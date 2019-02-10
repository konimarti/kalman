package kalman

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type filterRoseImpl struct {
	Ctx  context
	Pred prediction
	Upd  update
	Rose rose

	savedState *mat.VecDense
}

//Apply implements the Filter interface
func (f *filterRoseImpl) Apply(z, ctrl mat.Vector) mat.Vector {

	// update rose
	f.Upd.R = f.Rose.R(z)
	f.Pred.Q = f.Rose.Q(z, f.Ctx.X, f.Ctx.P, f.Pred.F, f.Upd.H, f.Upd.R)

	// correct state and covariance
	err := f.Upd.Update(&f.Ctx, z)
	if err != nil {
		fmt.Println(err)
	}

	// get y
	var filtered mat.VecDense
	filtered.MulVec(f.Upd.H, f.Ctx.X)

	// save state
	f.savedState = mat.VecDenseCopyOf(f.Ctx.X)

	// predict next state and covariance
	err = f.Pred.NextState(&f.Ctx, ctrl)
	if err != nil {
		fmt.Println(err)
	}
	err = f.Pred.NextCovariance(&f.Ctx)
	if err != nil {
		fmt.Println(err)
	}

	return &filtered
}

//State return the current state of the context
func (f *filterRoseImpl) State() mat.Vector {
	var state mat.VecDense
	state.CloneVec(f.savedState)
	return &state
}

type rose struct {
	Gamma  float64
	AlphaR float64
	AlphaM float64

	G *mat.Dense

	E1  *mat.VecDense
	EE1 *mat.Dense

	M *mat.Dense
}

//Q
func (r *rose) Q(y, x mat.Vector, P, F, H, R *mat.Dense) *mat.Dense {
	// dy = y - hx
	var hx, dy mat.VecDense
	hx.MulVec(H, x)
	dy.SubVec(y, &hx)

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
	q.Product(r.G, &qk, r.G.T())

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
//X: initial state
//P: initial covariance matrix
//F: prediction matrix
//B: control matrix
//H: scaling matrix for measurements
//G: discretized G matrix for system noise
//gammaR: Gain factor for measurement noise
//alphaR: Kalman gain for measurment covariance noise
//alphaM: Kalman gain for covariance M
func NewRoseFilter(X *mat.VecDense, P, F, B, H, G *mat.Dense, gammaR, alphaR, alphaM float64) Filter {
	var ctx context
	var pred prediction
	var upd update
	var rose rose

	// context
	ctx.X = X
	ctx.P = P

	// prediction
	pred.F = F
	pred.B = B

	// noises
	pred.Q = nil

	// update
	n, _ := H.Dims()
	upd.H = H
	upd.R = nil

	// rose
	rose.Gamma = gammaR
	rose.AlphaR = alphaR
	rose.AlphaM = alphaM
	rose.G = G
	rose.E1 = mat.NewVecDense(n, nil)
	rose.EE1 = mat.NewDense(n, n, nil)
	rose.M = mat.NewDense(n, n, nil)

	return &filterRoseImpl{ctx, pred, upd, rose, nil}
}
