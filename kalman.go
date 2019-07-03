package kalman

// https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

import (
	"fmt"

	"github.com/konimarti/lti"
	"gonum.org/v1/gonum/mat"
)

//Context contains the current state and covariance of the system
type Context struct {
	X *mat.VecDense // Current system state
	P *mat.Dense    // Current covariance matrix
}

//Noise represents the measurement and system noise 
type Noise struct {
	Q *mat.Dense // (discretized) system noise
	R *mat.Dense    // measurement noise
}

//Filter interface for using the Kalman filter
type Filter interface {
	Apply(z, ctrl *mat.VecDense) mat.Vector
	State() mat.Vector
}

//filtImpl is the implementation of the filter interface
type filterImpl struct {
	Ctx  Context
	Lti  lti.Discrete
	Nse  Noise
	savedState *mat.VecDense
}

//NewFilter returns a Kalman filter
func NewFilter(ctx Context, lti lti.Discrete, nse Noise) Filter {
	return &filterImpl{ctx, lti, nse, ctx.X}
}

//Apply implements the Filter interface
func (f *filterImpl) Apply(z, ctrl *mat.VecDense) mat.Vector {
	// correct state and covariance
	err := f.Update(z, ctrl)
	if err != nil {
		fmt.Println(err)
	}

	// get response of system y
	filtered := f.Lti.Response(f.Ctx.X, ctrl)

	// save current context state
	f.savedState = mat.VecDenseCopyOf(f.Ctx.X)

	// predict new state 
	f.NextState(ctrl)

	// predict new covariance matrix
	f.NextCovariance()

	return filtered
}

//NextState
func (f *filterImpl) NextState(ctrl *mat.VecDense) error {
	// X_k = Ad * X_k-1 + Bd * ctrl
	f.Ctx.X = f.Lti.Predict(f.Ctx.X, ctrl)
	return nil
}

//NextCovariance
func (f *filterImpl) NextCovariance() error {
	// P_new = Ad * P * Ad^t + Q
	f.Ctx.P.Product(f.Lti.Ad, f.Ctx.P, f.Lti.Ad.T())
	f.Ctx.P.Add(f.Ctx.P, f.Nse.Q)
	return nil
}


//Update performs Kalman update
func (f *filterImpl) Update(z, ctrl mat.Vector) error {
	// kalman gain
	// K = P H^T (H P H^T + R)^-1
	var K, kt, PHt, HPHt, denom mat.Dense
	PHt.Mul(f.Ctx.P, f.Lti.C.T())
	HPHt.Mul(f.Lti.C, &PHt)
	denom.Add(&HPHt, f.Nse.R)

	// calculation of Kalman gain with mat.Solve(..)
	// K = P H^T (H P H^T + R)^-1
	// K * (H P H^T + R) = P H^T
	// (H P H^T + R)^T K^T = (P H^T )^T
	err := kt.Solve(denom.T(), PHt.T())
	if err != nil {
		fmt.Println(err)
		fmt.Println("Setting Kalman gain to zero")
		denom.Zero()
		K.Product(f.Ctx.P, f.Lti.C.T(), &denom)
	} else {
		K.CloneFrom(kt.T())
	}

	// update state
	// X~_k = X_k + K * [z_k - H * X_k - D * ctrl ]
	var HXk, DCtrl, bracket, Kupd mat.VecDense
	HXk.MulVec(f.Lti.C, f.Ctx.X)
	DCtrl.MulVec(f.Lti.D, ctrl)
	bracket.SubVec(z, &HXk)
	//bracket.SubVec(&bracket, &DCtrl)
	Kupd.MulVec(&K, &bracket)
	f.Ctx.X.AddVec(f.Ctx.X, &Kupd)

	// update covariance
	// P~_k = P_k - K * [H_k * P_k]
	var KHP mat.Dense
	KHP.Product(&K, f.Lti.C, f.Ctx.P)
	f.Ctx.P.Sub(f.Ctx.P, &KHP)

	return nil
}

//State return the current state of the context
func (f *filterImpl) State() mat.Vector {
	var state mat.VecDense
	state.CloneVec(f.savedState)
	return &state
}

