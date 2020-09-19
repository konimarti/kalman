package kalman

import (
	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

type roseImpl struct {
	Std  filterImpl
	Rose rose
}

//Apply implements the Filter interface
func (r *roseImpl) Apply(ctx *Context, z, ctrl *mat.VecDense) mat.Vector {

	// adaptively update noise matrices
	r.Std.Nse.R = r.Rose.R(z)
	r.Std.Nse.Q = r.Rose.Q(ctx, z, ctrl, r.Std.Lti, r.Std.Nse.R)

	return r.Std.Apply(ctx, z, ctrl)
}

//State return the current state of the context
func (r *roseImpl) State() mat.Vector {
	var state mat.VecDense
	state.CloneFromVec(r.Std.savedState)
	return &state
}

type rose struct {
	Gamma  float64
	AlphaR float64
	AlphaM float64

	Gd *mat.Dense

	E1  *mat.VecDense
	EE1 *mat.Dense

	M *mat.Dense
}

//Q
func (r *rose) Q(ctx *Context, y, ctrl *mat.VecDense, lti lti.Discrete, R *mat.Dense) *mat.Dense {
	x := ctx.X
	P := ctx.P
	H := lti.C
	F := lti.Ad
	D := lti.D

	// dy = y - hx - d ctrl
	var hx, dctrl, dy mat.VecDense
	hx.MulVec(H, x)
	dctrl.MulVec(D, ctrl)
	dy.SubVec(y, &hx)
	dy.SubVec(&dy, &dctrl)

	// M = AlphaM * dy * dy' + (1-AlphaM) * M
	var dy2 mat.Dense
	dy2.Outer(r.AlphaM, &dy, &dy)
	r.M.Scale(1.0-r.AlphaM, r.M)
	r.M.Add(&dy2, r.M)

	// hmrh = H^T (M-R) H
	var mr, hmrh mat.Dense
	mr.Sub(r.M, R)
	hmrh.Product(H.T(), &mr, H)

	// fpf = F P F^T
	var fpf mat.Dense
	fpf.Product(F, P, F.T())

	// qk = hmrh - fpf = H^T (M-R) H - F P F^T
	var qk mat.Dense
	qk.Sub(&hmrh, &fpf)

	// q = Gd * qk * Gd^T
	var q mat.Dense
	q.Product(r.Gd, &qk, r.Gd.T())

	// make sure there are no negative values
	n, c := q.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < c; j++ {
			if q.At(i, j) < 0.0 {
				q.Set(i, j, 0.0)
			}
		}
	}

	return &q
}

//R returns a matrix with an adaptivly calculated measurement error
func (r *rose) R(y mat.Vector) *mat.Dense {
	// E1 = AlphaR * y(k) + (1-AlphaR) * E1
	var ay mat.VecDense
	ay.ScaleVec(r.AlphaR, y)
	r.E1.ScaleVec(1.0-r.AlphaR, r.E1)
	r.E1.AddVec(&ay, r.E1)

	// EE1 = AlphaR * (y(k) * y'(k)) + (1-AlphaR) * EE1
	var y2 mat.Dense
	y2.Outer(r.AlphaR, y, y)
	r.EE1.Scale(1.0-r.AlphaR, r.EE1)
	r.EE1.Add(&y2, r.EE1)

	// calculate R = Gamma * (EE1 - E1 * E1')
	var e1e1, diff, rm mat.Dense
	e1e1.Outer(1.0, r.E1, r.E1)
	diff.Sub(r.EE1, &e1e1)
	rm.Scale(r.Gamma, &diff)

	return &rm
}

//NewRoseFilter returns a ROSE Kalman filter
//Rapid Ongoing Stochasic covariance Estimation (ROSE) Filter
//lti: discrete linear, time-invariante system
//Gd: discretized G matrix for system noise
//gammaR: Gain factor for measurement noise
//alphaR: Kalman gain for measurment covariance noise
//alphaM: Kalman gain for covariance M
func NewRoseFilter(lti lti.Discrete, Gd *mat.Dense, gammaR, alphaR, alphaM float64) Filter {
	// create dummy noise struct
	q, _ := lti.Ad.Dims()
	r, _ := lti.C.Dims()
	nse := NewZeroNoise(q, r)

	// create the standard Kalman filter
	filter := filterImpl{lti, nse, nil}

	// create new rose struct
	rose := rose{
		Gamma:  gammaR,
		AlphaR: alphaR,
		AlphaM: alphaM,
		Gd:     Gd,
		E1:     mat.NewVecDense(r, nil),
		EE1:    mat.NewDense(r, r, nil),
		M:      mat.NewDense(r, r, nil),
	}

	return &roseImpl{filter, rose}
}
