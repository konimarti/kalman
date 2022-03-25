package kalman

// https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

import (
	"fmt"

	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

//Context contains the current state and covariance of the system
type Context struct {
	X                         *mat.VecDense // Current system state
	P                         *mat.Dense    // Current covariance matrix
	pmt                       mat.Dense     // Workspace for calculating next covariance
	mpmt                      mat.Dense     // Workspace for calculating next covariance
	k, kt, pct, cpct, denom   mat.Dense     // Workspace for calculating Update()
	cxk, dCtrl, bracket, kupd mat.VecDense
	kcp                       mat.Dense
}

//Noise represents the measurement and system noise
type Noise struct {
	Q *mat.Dense // (discretized) system noise
	R *mat.Dense // measurement noise
}

//NewZeroNoise initializes a Noise struct
//q: dimension of square matrix Q
//r: dimension of square matrix R
func NewZeroNoise(q, r int) Noise {
	nse := Noise{
		Q: mat.NewDense(q, q, nil),
		R: mat.NewDense(r, r, nil),
	}
	return nse
}

//Filter interface for using the Kalman filter
type Filter interface {
	Apply(ctx *Context, z, ctrl *mat.VecDense) mat.Vector
	State() mat.Vector
}

//filtImpl is the implementation of the filter interface
type filterImpl struct {
	Lti        lti.Discrete
	Nse        Noise
	savedState *mat.VecDense
}

//NewFilter returns a Kalman filter
func NewFilter(lti lti.Discrete, nse Noise) Filter {
	return &filterImpl{lti, nse, nil}
}

//Apply implements the Filter interface
func (f *filterImpl) Apply(ctx *Context, z, ctrl *mat.VecDense) mat.Vector {
	// correct state and covariance
	err := f.Update(ctx, z, ctrl)
	if err != nil {
		fmt.Println(err)
	}

	// get response of system y
	filtered := f.Lti.Response(ctx.X, ctrl)

	// save current context state
	f.savedState = mat.VecDenseCopyOf(ctx.X)

	// predict new state
	f.NextState(ctx, ctrl)

	// predict new covariance matrix
	f.NextCovariance(ctx)

	return filtered
}

//NextState
func (f *filterImpl) NextState(ctx *Context, ctrl *mat.VecDense) error {
	// X_k = Ad * X_k-1 + Bd * ctrl
	ctx.X = f.Lti.Predict(ctx.X, ctrl)
	return nil
}

//NextCovariance
func (f *filterImpl) NextCovariance(ctx *Context) error {
	// P_new = Ad * P * Ad^t + Q
	//ctx.P.Product(f.Lti.Ad, ctx.P, f.Lti.Ad.T())
	//ctx.P.Add(ctx.P, f.Nse.Q)
	ctx.P = lti.NewCovariance(f.Lti.Ad).Predict(ctx.P, f.Nse.Q, &ctx.pmt, &ctx.mpmt)
	return nil
}

//Update performs Kalman update
func (f *filterImpl) Update(ctx *Context, z, ctrl mat.Vector) error {
	// kalman gain
	// K = P C^T (C P C^T + R)^-1
	ctx.pct.Mul(ctx.P, f.Lti.C.T())
	ctx.cpct.Mul(f.Lti.C, &ctx.pct)
	ctx.denom.Add(&ctx.cpct, f.Nse.R)

	// calculation of Kalman gain with mat.Solve(..)
	// K = P C^T (C P C^T + R)^-1
	// K * (C P C^T + R) = P C^T
	// (C P C^T + R)^T K^T = (P C^T )^T
	err := ctx.kt.Solve(ctx.denom.T(), ctx.pct.T())
	if err != nil {
		//log.Println(err)
		//log.Println("setting Kalman gain to zero")
		ctx.denom.Zero()
		ctx.k.Product(ctx.P, f.Lti.C.T(), &ctx.denom)
	} else {
		r, c := ctx.k.Dims()
		if r == 0 && c == 0 {
			ctx.k.CloneFrom(ctx.kt.T())
		} else {
			ctx.k.Copy(ctx.kt.T())
		}
	}

	// update state
	// X~_k = X_k + K * [z_k - C * X_k - D * ctrl ]
	ctx.cxk.MulVec(f.Lti.C, ctx.X)
	ctx.dCtrl.MulVec(f.Lti.D, ctrl)
	ctx.bracket.SubVec(z, &ctx.cxk)
	ctx.bracket.SubVec(&ctx.bracket, &ctx.dCtrl)
	ctx.kupd.MulVec(&ctx.k, &ctx.bracket)
	ctx.X.AddVec(ctx.X, &ctx.kupd)

	// update covariance
	// P~_k = P_k - K * [C * P_k]
	ctx.kcp.Product(&ctx.k, f.Lti.C, ctx.P)
	ctx.P.Sub(ctx.P, &ctx.kcp)

	return nil
}

//State return the current state of the context
func (f *filterImpl) State() mat.Vector {
	var state mat.VecDense
	state.CloneFromVec(f.savedState)
	return &state
}
