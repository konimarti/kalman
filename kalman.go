package kalman

// https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

import (
	"gonum.org/v1/gonum/mat"
)

type context struct {
	X *mat.VecDense // Current state
	P *mat.Dense    // Current covariance matrix
}

type prediction struct {
	F *mat.Dense // Prediction matrix
	B *mat.Dense // Control matrix

	//W *mat.Dense // Prediction noise
	Q *mat.Dense // External noise
}

func (P *prediction) NextState(ctx *context, ctrl mat.Vector) error {
	// predict new state
	// X_k = F * X_k-1 + B * ctrl + W
	var Fx, Bmu mat.VecDense
	Fx.MulVec(P.F, ctx.X)
	Bmu.MulVec(P.B, ctrl)
	ctx.X.AddVec(&Fx, &Bmu)
	return nil
}

func (P *prediction) NextCovariance(ctx *context) error {
	// predict new covariance matrix
	// P_new = F * P * F^t + Q
	var PFt mat.Dense
	PFt.Mul(ctx.P, P.F.T())
	ctx.P.Mul(P.F, &PFt)
	ctx.P.Add(ctx.P, P.Q)
	return nil
}

type update struct {
	H *mat.Dense // scaling matrix
	R *mat.Dense // measurements errors
}

func (u *update) Update(ctx *context, z mat.Vector) error {
	// kalman gain
	var K, PHt, HPHt, denom mat.Dense
	PHt.Mul(ctx.P, u.H.T())
	HPHt.Mul(u.H, &PHt)
	denom.Add(&HPHt, u.R)
	if err := denom.Inverse(&denom); err != nil {
		panic(err)
	}
	K.Mul(&PHt, &denom)

	// update state
	// X'_k = X_k + K * [z_k - H * X_k]
	var HXk, bracket, Kupd mat.VecDense
	HXk.MulVec(u.H, ctx.X)
	bracket.SubVec(z, &HXk)
	Kupd.MulVec(&K, &bracket)
	ctx.X.AddVec(ctx.X, &Kupd)

	// update covariance
	// P'_k = P_k - K' * [H_k * P_k]
	var KHP, HP mat.Dense
	HP.Mul(u.H, ctx.P)
	KHP.Mul(&K, &HP)
	ctx.P.Sub(ctx.P, &KHP)

	return nil
}

//Filter interface for using the Kalman filter
type Filter interface {
	Apply(z, ctrl mat.Vector) mat.Vector
	State() mat.Vector
}

type filterImpl struct {
	Ctx  context
	Pred prediction
	Upd  update
}

//Apply implements the Filter interface
func (f *filterImpl) Apply(z, ctrl mat.Vector) mat.Vector {
	f.Pred.NextState(&f.Ctx, ctrl)
	f.Pred.NextCovariance(&f.Ctx)
	f.Upd.Update(&f.Ctx, z)
	var filtered mat.VecDense
	filtered.MulVec(f.Upd.H, f.Ctx.X)
	return &filtered
}

//State return the current state of the context
func (f *filterImpl) State() mat.Vector {
	var state mat.VecDense
	state.CloneVec(f.Ctx.X)
	return &state
}

//NewFilter returns a Kalman filter
//X: initial state
//P: initial covariance matrix
//F: prediction matrix
//B: control matrix
//H: scaling matrix for measurements
//R: measurement error matrix
func NewFilter(X *mat.VecDense, P, F, B, Q, H, R *mat.Dense) Filter {
	var ctx context
	var pred prediction
	var upd update

	// context
	ctx.X = X
	ctx.P = P

	// prediction
	pred.F = F
	pred.B = B

	// noises
	//pred.W = mat.NewDense(2, 2, nil)
	pred.Q = Q

	// update
	upd.H = H
	upd.R = R

	return &filterImpl{ctx, pred, upd}
}
