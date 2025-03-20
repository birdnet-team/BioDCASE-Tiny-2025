# automatically generated by the FlatBuffers compiler, do not modify

# namespace: FeatureConfigs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FilterbankConfig(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FilterbankConfig()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFilterbankConfig(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # FilterbankConfig
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FilterbankConfig
    def FftStartIdx(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # FilterbankConfig
    def FftEndIdx(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # FilterbankConfig
    def Weights(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FilterbankConfig
    def WeightsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FilterbankConfig
    def WeightsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FilterbankConfig
    def WeightsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # FilterbankConfig
    def Unweights(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FilterbankConfig
    def UnweightsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FilterbankConfig
    def UnweightsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FilterbankConfig
    def UnweightsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # FilterbankConfig
    def NumChannels(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FilterbankConfig
    def ChannelFrequencyStarts(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FilterbankConfig
    def ChannelFrequencyStartsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FilterbankConfig
    def ChannelFrequencyStartsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FilterbankConfig
    def ChannelFrequencyStartsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # FilterbankConfig
    def ChannelWeightStarts(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FilterbankConfig
    def ChannelWeightStartsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FilterbankConfig
    def ChannelWeightStartsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FilterbankConfig
    def ChannelWeightStartsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # FilterbankConfig
    def ChannelWidths(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FilterbankConfig
    def ChannelWidthsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FilterbankConfig
    def ChannelWidthsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FilterbankConfig
    def ChannelWidthsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

def FilterbankConfigStart(builder):
    builder.StartObject(8)

def FilterbankConfigAddFftStartIdx(builder, fftStartIdx):
    builder.PrependUint32Slot(0, fftStartIdx, 0)

def FilterbankConfigAddFftEndIdx(builder, fftEndIdx):
    builder.PrependUint32Slot(1, fftEndIdx, 0)

def FilterbankConfigAddWeights(builder, weights):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(weights), 0)

def FilterbankConfigStartWeightsVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FilterbankConfigAddUnweights(builder, unweights):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(unweights), 0)

def FilterbankConfigStartUnweightsVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FilterbankConfigAddNumChannels(builder, numChannels):
    builder.PrependInt32Slot(4, numChannels, 0)

def FilterbankConfigAddChannelFrequencyStarts(builder, channelFrequencyStarts):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(channelFrequencyStarts), 0)

def FilterbankConfigStartChannelFrequencyStartsVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FilterbankConfigAddChannelWeightStarts(builder, channelWeightStarts):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(channelWeightStarts), 0)

def FilterbankConfigStartChannelWeightStartsVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FilterbankConfigAddChannelWidths(builder, channelWidths):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(channelWidths), 0)

def FilterbankConfigStartChannelWidthsVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FilterbankConfigEnd(builder):
    return builder.EndObject()



class FeatureConfig(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FeatureConfig()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsFeatureConfig(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # FeatureConfig
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FeatureConfig
    def HanningWindow(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FeatureConfig
    def HanningWindowAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FeatureConfig
    def HanningWindowLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FeatureConfig
    def HanningWindowIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # FeatureConfig
    def WindowScalingBits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # FeatureConfig
    def FftTwiddle(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int16Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 2))
        return 0

    # FeatureConfig
    def FftTwiddleAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int16Flags, o)
        return 0

    # FeatureConfig
    def FftTwiddleLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # FeatureConfig
    def FftTwiddleIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # FeatureConfig
    def FbConfig(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = FilterbankConfig()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # FeatureConfig
    def MelRangeMin(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FeatureConfig
    def MelRangeMax(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # FeatureConfig
    def MelPostScalingBits(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

def FeatureConfigStart(builder):
    builder.StartObject(7)

def FeatureConfigAddHanningWindow(builder, hanningWindow):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(hanningWindow), 0)

def FeatureConfigStartHanningWindowVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FeatureConfigAddWindowScalingBits(builder, windowScalingBits):
    builder.PrependUint8Slot(1, windowScalingBits, 0)

def FeatureConfigAddFftTwiddle(builder, fftTwiddle):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(fftTwiddle), 0)

def FeatureConfigStartFftTwiddleVector(builder, numElems):
    return builder.StartVector(2, numElems, 2)

def FeatureConfigAddFbConfig(builder, fbConfig):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(fbConfig), 0)

def FeatureConfigAddMelRangeMin(builder, melRangeMin):
    builder.PrependInt32Slot(4, melRangeMin, 0)

def FeatureConfigAddMelRangeMax(builder, melRangeMax):
    builder.PrependInt32Slot(5, melRangeMax, 0)

def FeatureConfigAddMelPostScalingBits(builder, melPostScalingBits):
    builder.PrependUint8Slot(6, melPostScalingBits, 0)

def FeatureConfigEnd(builder):
    return builder.EndObject()



