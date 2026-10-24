namespace QENN {
    open Microsoft.Quantum.Math;

    // Classical amplitude function used by the QENN hidden units: [cos(e^theta), sin(e^theta)] with theta clamped to [-10, 10].
    // NOTE: the original file called ExpD, which does not exist in the current Q# standard library; E() ^ x is the supported form.
    function GetAmplitudes(theta : Double) : Double[] {
        let clipped = MaxD(-10.0, MinD(10.0, theta));
        let e = E() ^ clipped;
        return [Cos(e), Sin(e)];
    }
}
