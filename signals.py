from enum import Enum
from math import sin, pi

import numpy as np

CORRECTION = 0.02


class SignalType(Enum):
    Sine = 0
    Pulse = 1
    Triangular = 2
    Sawtooth = 3
    Noise = 4

    @classmethod
    def from_string(cls, string):
        mappings = {
            'sine': cls.Sine,
            'pulse': cls.Pulse,
            'triangular': cls.Triangular,
            'sawtooth': cls.Sawtooth,
            'noise': cls.Noise
        }
        return mappings.get(string.lower())


def signal(signal_type, sample_rate, params):
    if signal_type == SignalType.Sine:
        return lambda i: sine_signal(i, sample_rate, params['amplitude'], params['frequency'], params['phase_shift'])
    if signal_type == SignalType.Pulse:
        return lambda i: pulse_signal(i, sample_rate, params['amplitude'], params['frequency'],
                                      params['phase_shift'], params['duty_cycle_threshold'])
    if signal_type == SignalType.Triangular:
        return lambda i: triangular_signal(i, sample_rate, params['amplitude'], params['frequency'], params['phase_shift'])
    if signal_type == SignalType.Sawtooth:
        return lambda i: sawtooth_signal(i, sample_rate, params['amplitude'], params['frequency'], params['phase_shift'])
    if signal_type == SignalType.Noise:
        return noise_signal(sample_rate)


def signal_for_modulation(signal_type, sample_rate, params):
    if signal_type == SignalType.Sine:
        return lambda i, amplitude, frequency, phase_accumulator=None: \
            sine_signal(i, sample_rate, amplitude, frequency, params['phase_shift'], phase_accumulator)
    if signal_type == SignalType.Pulse:
        return lambda i, amplitude, frequency, phase_accumulator=None: \
            pulse_signal(i, sample_rate, amplitude, frequency,
                         params['phase_shift'], params['duty_cycle_threshold'], phase_accumulator)
    if signal_type == SignalType.Triangular:
        return lambda i, amplitude, frequency, phase_accumulator=None: \
            triangular_signal(i, sample_rate, amplitude, frequency, params['phase_shift'], phase_accumulator)
    if signal_type == SignalType.Sawtooth:
        return lambda i, amplitude, frequency, phase_accumulator=None: \
            sawtooth_signal(i, sample_rate, amplitude, frequency, params['phase_shift'], phase_accumulator)


def sine_signal(i, sample_rate, amplitude, frequency, phase_shift, phase_accumulator=None):
    phase = 2 * pi * frequency * i / sample_rate + phase_shift
    if phase_accumulator is not None:
        phase = phase_accumulator[0] = phase_accumulator[0] + phase
    return amplitude / 2 * (1 + sin(phase))


def pulse_signal(i, sample_rate, amplitude, frequency, phase_shift, duty_cycle_threshold, phase_accumulator=None):
    phase = 2 * pi * frequency * i / sample_rate + phase_shift
    if phase_accumulator is not None:
        phase = phase_accumulator[0] = phase_accumulator[0] + phase
    if amplitude * sin(phase) > duty_cycle_threshold:
        return amplitude
    else:
        return 0.0


def triangular_signal(i, sample_rate, amplitude, frequency, phase_shift, phase_accumulator=None):
    phase = 4 * pi * frequency * i / sample_rate + phase_shift
    if phase_accumulator is not None:
        phase = phase_accumulator[0] = phase_accumulator[0] + phase
    arg = abs(phase) % (4 * pi)
    if arg < (2 * pi):
        return amplitude * arg / (2 * pi)
    else:
        return amplitude * (4 * pi - arg) / (2 * pi)


def sawtooth_signal(i, sample_rate, amplitude, frequency, phase_shift, phase_accumulator=None):
    phase = 2 * pi * frequency * i / sample_rate + phase_shift
    if phase_accumulator is not None:
        phase = phase_accumulator[0] = phase_accumulator[0] + phase
    return amplitude * (abs(phase) % (2 * pi - CORRECTION)) / (2 * pi)


def noise_signal(period):
    values = tuple(np.random.normal(0, 1, period))
    return lambda i: values[i % len(values)]


def aggregate_to_polyharmonic_signal(signals):
    return lambda i: sum(s(i) for s in signals)
