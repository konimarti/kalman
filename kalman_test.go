package kalman

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func NewSetup() (*context, *prediction, *update) {
	var ctx context
	var pred prediction
	var upd update

	// init ctx
	ctx.X = mat.NewVecDense(2, []float64{4000.0, 280.0})
	ctx.P = mat.NewDense(2, 2, []float64{425, 0, 0, 25})

	// init prediction
	dt := 1.0
	pred.F = mat.NewDense(2, 2, []float64{1, dt, 0, 1})
	pred.B = mat.NewDense(2, 1, []float64{0.5 * dt * dt, dt})

	//pred.W = mat.NewDense(2, 2, nil)
	pred.Q = mat.NewDense(2, 2, nil)

	// init update
	upd.H = mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	upd.R = mat.NewDense(2, 2, []float64{625, 0, 0, 36})

	return &ctx, &pred, &upd
}

func TestPredictionState(t *testing.T) {
	ctx, pred, _ := NewSetup()

	// predict next state
	ctrl := mat.NewVecDense(1, []float64{2.0})
	if err := pred.NextState(ctx, ctrl); err != nil {
		t.Error(err)
	}
	expectedVec := mat.NewVecDense(2, []float64{4281.0, 282.0})
	if !mat.EqualApprox(expectedVec, ctx.X, 1e-4) {
		fmt.Println("actual:", ctx.X)
		fmt.Println("expected:", expectedVec)
		t.Error("PredictState")
	}
}

func TestPredictionCovariance(t *testing.T) {
	ctx, pred, _ := NewSetup()

	// predict next covariance
	if err := pred.NextCovariance(ctx); err != nil {
		t.Error(err)
	}
	expected := mat.NewDense(2, 2, []float64{450.0, 25.0, 25.0, 25.0})
	if !mat.EqualApprox(expected, ctx.P, 1e-4) {
		fmt.Println("actual:", ctx.P)
		fmt.Println("expected:", expected)
		t.Error("PredictCovariance")
	}
}

func TestUpdate(t *testing.T) {
	ctx, pred, upd := NewSetup()

	// predict next state
	ctrl := mat.NewVecDense(1, []float64{2.0})
	if err := pred.NextState(ctx, ctrl); err != nil {
		t.Error(err)
	}
	// predict next covariance
	if err := pred.NextCovariance(ctx); err != nil {
		t.Error(err)
	}

	// update
	z := mat.NewVecDense(2, []float64{4260.0, 282.0})
	if err := upd.Update(ctx, z); err != nil {
		t.Error(err)
	}
	expectedX := mat.NewVecDense(2, []float64{4272.32678984, 281.70900693})
	if !mat.EqualApprox(expectedX, ctx.X, 1e-4) {
		fmt.Println("actual:", ctx.X)
		fmt.Println("expected:", expectedX)
		t.Error("UpdateState")
	}
	expectedP := mat.NewDense(2, 2, []float64{258.1312548113933, 8.660508083140876, 8.66050808314088, 14.549653580})
	if !mat.EqualApprox(expectedP, ctx.P, 1e-4) {
		fmt.Println("actual:", ctx.P)
		fmt.Println("expected:", expectedP)
		t.Error("UpdateCovariance")
	}
}

func TestFilter(t *testing.T) {
	ctx, pred, upd := NewSetup()
	filter := NewFilter(ctx.X, ctx.P, pred.F, pred.B, pred.Q, upd.H, upd.R)

	ctrl := mat.NewVecDense(1, []float64{2.0})
	z := mat.NewVecDense(2, []float64{4260.0, 282.0})

	filteredState := filter.Apply(z, ctrl)
	expectedX := mat.NewVecDense(2, []float64{4272.32678984, 281.70900693})
	if !mat.EqualApprox(expectedX, filteredState, 1e-4) {
		fmt.Println("actual:", filteredState)
		fmt.Println("expected:", expectedX)
		t.Error("ApplyFilter")
	}

	checkState := filter.State()
	if !mat.EqualApprox(filteredState, checkState, 1e-4) {
		fmt.Println("actual:", checkState)
		fmt.Println("expected:", filteredState)
		t.Error("StateFilter")
	}

}
