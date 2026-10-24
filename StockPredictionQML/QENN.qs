namespace QENN {
    open Microsoft.Quantum.Math;

    function GetAmplitudes(theta : Double) : Double[] {
        let clipped = MaxD(-10.0, MinD(10.0, theta));
        let e = ExpD(clipped);
        return [Cos(e), Sin(e)];
    }
}
