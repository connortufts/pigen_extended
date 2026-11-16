"""

This file defines the general memristor solver model base class

Each memristor device implements the required functions here to interface with the model.
Per Tung 2026, the internal state of every memristor follows:

ds / dt = F_fwd + F_rev

F_fwd = (s_fwd - s) / Tau_fwd
F_rev = (s_rev - s) / Tau_rev

s represents the state of the device
    For example, the polarization of a FeFET device or filament length of an RRAM device

F_fwd represents the function for driving the device to the forward state
    s_fwd is the final forward state of the device
    Tau_fwd is the effective time constant of the switching
The same definitions apply to _rev variables (the reverse state)

"""

import math
import torch

from core.physics import DevicePhysics

class UnifiedModel(DevicePhysics):
    #==============================
    # How the solver is organized, outermost to innermost:
    #
    #   simulate()                one sequence           loops over the dataset samples
    #     advanceInterval()       one sample interval    loops over substeps until the sample interval is covered
    #       solveSubstep()        one substep            gets tau from timeConstants(), then solves exactly
    #         timeConstants()     the device physics     tauFwd, tauRev, stateFwd, stateRev at the present state
    #
    # sampleInterval is the time between two dataset samples; substepLength is a piece of one sample interval
    # A device subclass writes params(), initialState(), timeConstants() and, if it has aux states, updateAux()
    # simulate(), advanceInterval() and solveSubstep() are shared by every device and are not overridden

    # Largest state change per accepted substep, as a fraction of |stateFwd - stateRev|.
    MAX_DS_DELTA = 0.02
    # Smallest substep is sampleInterval / MAX_SUBSTEPS.
    MAX_SUBSTEPS = 4096

    #==============================
    # Returns deviceParams, the constants the device needs for one sequence (any object, e.g. a dict of parameters)
    # Called once per sequence by simulate(), before any substep
    # deviceParams is passed back unchanged to initialState(), timeConstants() and updateAux()
    # voltageWaveform is the whole sequence's voltage [V], so polarity-dependent constants can be chosen here
    def params(self, material, voltageWaveform):
        raise NotImplementedError

    #==============================
    # Returns tuple of (initialState, initialAuxStates)
    # These are the values used for the initial state of the device
    # initialState is the starting state, in physical units
    # initialAuxStates is a tuple of any other required initial state values specific to the device
    # voltageWaveform is the whole sequence's voltage [V], e.g. to pick the starting state from polarity()
    def initialState(self, voltageWaveform, deviceParams):
        raise NotImplementedError

    #==============================
    # Returns tuple of (tauFwd, tauRev, stateFwd, stateRev)
    # These define the switching at the present state, voltage [V] and current [A]
    # tauFwd and tauRev are the effective time constants [s], must be > 0
    #   During forward switching tauFwd is active and tauRev becomes inf, and vice versa
    #   Return math.inf for a direction that is switched off
    # stateFwd and stateRev are the final states of forward and reverse switching
    # This is where each device's physics goes
    def timeConstants(self, state, voltage, current, auxStates, deviceParams):
        raise NotImplementedError

    #==============================
    # Returns the updated auxStates tuple after a substep of length substepLength that ended at state
    # Called by solveSubstep() for every substep; advanceInterval() keeps the result only if the substep is accepted
    # Default leaves auxStates unchanged, for devices with no auxiliary states
    def updateAux(self, state, voltage, current, auxStates, substepLength, deviceParams):
        return auxStates

    #==============================
    # Returns tuple of (newState, newAuxStates, stateRange)
    # Solves ONE substep of length substepLength, holding tauFwd and tauRev constant over it
    # Called by advanceInterval(), which decides whether to accept the result
    # With constant time constants, the general equation becomes ds / dt = (stateTarget - s) / tauEff,
    # which is solved exactly:
    #   1 / tauEff         = 1 / tauFwd + 1 / tauRev
    #   stateTarget        = tauEff * (stateFwd / tauFwd + stateRev / tauRev)
    #   newState           = stateTarget + (state - stateTarget) * exp(-substepLength / tauEff)
    # stateTarget is the state the device is heading toward, always between stateRev and stateFwd,
    # so the state never leaves that range
    # This holds for any substepLength, even one much longer than tauEff (exponential Euler)
    # If both tau are inf, the state does not change
    # The only error is that tau really changes as the state moves; advanceInterval() limits that with short substeps
    # stateRange is |stateFwd - stateRev|, used by advanceInterval() to judge how far the state moved
    def solveSubstep(self, state, voltage, current, auxStates, substepLength, deviceParams):
        tauFwd, tauRev, stateFwd, stateRev = self.timeConstants(state, voltage, current, auxStates, deviceParams)
        inverseTauEff = 1.0 / tauFwd + 1.0 / tauRev     # summed as 1/tau so tau = inf contributes exactly 0
        if inverseTauEff > 0.0:
            stateTarget = (stateFwd / tauFwd + stateRev / tauRev) / inverseTauEff
            state = stateTarget + (state - stateTarget) * math.exp(-substepLength * inverseTauEff)
        newAuxStates = self.updateAux(state, voltage, current, auxStates, substepLength, deviceParams)
        return state, newAuxStates, abs(stateFwd - stateRev)

    #==============================
    # Returns tuple of (state, auxStates) at the end of one sample interval (the time between two dataset samples)
    # Called by simulate() once per sample; calls solveSubstep() one or more times
    # solveSubstep() is only exact while tauFwd and tauRev stay constant, but they usually change with the state
    # So the interval is split into substeps wherever the state moves fast:
    #   First try one substep covering the whole interval
    #   If a substep moves the state by more than MAX_DS_DELTA of the full range, it is rejected and halved
    #   If it is accepted, tau is re-evaluated at the new state on the next try, and the substep is doubled
    #   A substep never gets shorter than sampleInterval / MAX_SUBSTEPS, so the loop always ends
    # Where the state barely moves (before switching or after saturation) the first try is accepted: one substep
    # During switching one interval can take dozens of substeps
    def advanceInterval(self, state, voltage, current, auxStates, sampleInterval, deviceParams):
        remainingTime = sampleInterval
        substepLength = sampleInterval
        minSubstepLength = sampleInterval / self.MAX_SUBSTEPS
        while remainingTime > 0.0:
            substepLength = min(substepLength, remainingTime)
            newState, newAuxStates, stateRange = self.solveSubstep(
                state, voltage, current, auxStates, substepLength, deviceParams)
            movedTooFar = abs(newState - state) > self.MAX_DS_DELTA * stateRange
            if movedTooFar and substepLength > minSubstepLength:
                substepLength *= 0.5
                continue
            state, auxStates = newState, newAuxStates
            remainingTime -= substepLength
            substepLength *= 2.0
        return state, auxStates

    #==============================
    # Returns the physical state at every sample of one sequence, as a tensor
    # timeSteps, voltageWaveform and currentWaveform are 1-D tensors of the same length, in physical units
    #   timeSteps[n] is the time between sample n - 1 and sample n [s]
    # Steps:
    #   1. params() and initialState() set up the sequence
    #   2. advanceInterval() carries the state from each sample to the next
    #   3. the states are collected into a tensor
    # Sample 0 is the initial state; sample n is reached by holding voltage and current of sample n over timeSteps[n]
    # The output is the physical state; core normalizes it to [-1, 1] separately with normalize_state()
    # The math runs on Python floats in double precision:
    #   tau values span hundreds of orders of magnitude, which float32 cannot hold
    #   no gradients are needed, since the result is a fixed training target cached by core/cache.py
    # The output has the same dtype and device as the input
    def simulate(self, timeSteps, voltageWaveform, currentWaveform, material):
        deviceParams = self.params(material, voltageWaveform)
        timeStepList = timeSteps.detach().double().cpu().tolist()
        voltageList = voltageWaveform.detach().double().cpu().tolist()
        currentList = currentWaveform.detach().double().cpu().tolist()
        state, auxStates = self.initialState(voltageWaveform, deviceParams)
        states = [state]
        for sampleIndex in range(1, len(timeStepList)):
            state, auxStates = self.advanceInterval(
                state, voltageList[sampleIndex], currentList[sampleIndex], auxStates,
                timeStepList[sampleIndex], deviceParams)
            states.append(state)
        return torch.tensor(states, dtype=timeSteps.dtype, device=timeSteps.device)

    #==============================
    # Returns +1 for a forward pulse or -1 for a reverse pulse
    # Uses the sign of the largest-magnitude voltage, matching how core/data.py labels sequences
    # Devices use this in initialState() to start each pulse from the opposite final state
    @staticmethod
    def polarity(voltageWaveform):
        peakIndex = voltageWaveform.abs().argmax()
        return 1.0 if float(voltageWaveform[peakIndex]) > 0 else -1.0
