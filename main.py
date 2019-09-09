import json
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from signals import signal, SignalType, aggregate_to_polyharmonic_signal, signal_for_modulation

VALID_ARGC = 2


def main():
    if len(sys.argv) < VALID_ARGC:
        print('Usage: data_filepath (single|polyharmonic|modulation)')

    data_path = sys.argv[1]
    selected_task = sys.argv[2]
    with open(data_path, encoding='utf-8') as data_file:
        data = json.loads(data_file.read())

    task_solutions = {
        'single': lambda: demonstrate_single(data['sample_rate'], data['single']),
        'polyharmonic': lambda: demonstrate_polyharmonic(data['sample_rate'], data['polyharmonic']),
        'modulation': lambda: demonstrate_modulation(data['sample_rate'], data['single'], data['modulating'])
    }

    widgets = task_solutions[selected_task]()
    plt.show()


def demonstrate_single(sample_rate, signal_params):
    signal_type = SignalType.from_string(signal_params['signal'])
    analog_signal = signal(signal_type, sample_rate, signal_params)

    widgets = demonstrate_signal(sample_rate, analog_signal, f'{signal_params["signal"].title()} Signal')
    return widgets


def demonstrate_polyharmonic(sample_rate, params_of_signals):
    analog_signals = []
    for signal_params in params_of_signals:
        signal_type = SignalType.from_string(signal_params['signal'])
        analog_signals.append(signal(signal_type, sample_rate, signal_params))
    polyharmonic_analog_signal = aggregate_to_polyharmonic_signal(analog_signals)

    widgets = demonstrate_signal(sample_rate, polyharmonic_analog_signal, 'Polyharmonic Signal')
    return widgets


def demonstrate_modulation(sample_rate, signal_params, modulating_signal_params):
    fig, axes = plt.subplots(2, sharex=True)
    fig.canvas.set_window_title('Advanced Signals Modelling')
    fig.suptitle('Amplitude & Frequency Modulation')

    n = range(sample_rate)

    modulating_signal_type = SignalType.from_string(modulating_signal_params['signal'])
    modulating_analog_signal = signal(modulating_signal_type, sample_rate, modulating_signal_params)
    sampled_modulating_signal = tuple(map(modulating_analog_signal, n))

    signal_type = SignalType.from_string(signal_params['signal'])
    analog_signal = signal_for_modulation(signal_type, sample_rate, signal_params)

    sampled_signal_with_amplitude_modulation = tuple(analog_signal(i, sampled_modulating_signal[i],
                                                                   signal_params['frequency']) for i in n)

    accumulator = [0]
    sampled_signal_with_frequency_modulation = \
        tuple(analog_signal(1, signal_params['amplitude'],
                            signal_params['frequency'] + signal_params['frequency'] * sampled_modulating_signal[i], accumulator) for i in n)

    am_soundax = plt.axes([0.79, 0.54, 0.1, 0.04])
    fm_soundax = plt.axes([0.79, 0.12, 0.1, 0.04])

    widgets = [
        show_signal_modulation(axes[0], am_soundax, sample_rate, n, sampled_signal_with_amplitude_modulation,
                               sampled_modulating_signal, 'AM'),
        show_signal_modulation(axes[1], fm_soundax, sample_rate, n, sampled_signal_with_frequency_modulation,
                               sampled_modulating_signal, 'FM')
    ]

    return widgets


def demonstrate_signal(sample_rate, analog_signal, suptitle):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Advanced Signals Modelling')
    fig.suptitle(suptitle)
    ax.set_ylabel('x(n)')
    ax.set_xlabel('n')

    n = range(sample_rate)
    sampled_signal = tuple(map(analog_signal, n))
    ax.plot(n, sampled_signal)

    soundax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = add_play_button(sampled_signal, sample_rate, soundax)

    return [button]


def show_signal_modulation(ax, soundax, sample_rate, n, sampled_signal, sampled_modulating_signal, signal_label):
    ax.plot(n, sampled_signal, label=signal_label)
    ax.plot(n, sampled_modulating_signal, label='Modulating Signal')
    ax.legend(loc=1, prop={'size': 7})

    return add_play_button(sampled_signal, sample_rate, soundax)


def add_play_button(sampled_signal, sample_rate, soundax):
    def play_signal(_):
        sd.play(np.array(sampled_signal), sample_rate)
    button = Button(soundax, 'Play')
    button.on_clicked(play_signal)
    return button


if __name__ == '__main__':
    main()
