package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/konimarti/kalman"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// prepare output file
	file, err := os.Create("car.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	fmt.Fprintln(file, "Measured_v_x,Measured_v_y,Filtered_v_x,Filtered_v_y")

	// init state: pos_x = 0, pox_y = 0, v_x = 30 km/h, v_y = 10 km/h
	X := mat.NewVecDense(4, []float64{0, 0, 30, 10})
	// initial covariance matrix
	P := mat.NewDense(4, 4, []float64{
		1000, 0, 0, 0,
		0, 1000, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000})

	// prediction matrix
	dt := 0.1
	F := mat.NewDense(4, 4, []float64{
		1, 0, dt, 0,
		0, 1, 0, dt,
		0, 0, 1, 0,
		0, 0, 0, 1})

	// no external influence
	B := mat.NewDense(4, 4, nil)

	// process model covariance matrix
	Q := mat.NewDense(4, 4, []float64{
		dt * dt * dt * 0.33, 0, dt * dt * 0.5, 0,
		0, dt * dt * dt * 0.33, 0, dt * dt * 0.5,
		dt * dt * 0.5, 0, dt, 0,
		0, dt * dt * 0.5, 0, dt})

	// scaling matrix: only measure velocities
	H := mat.NewDense(2, 4, []float64{
		0, 0, 1, 0,
		0, 0, 0, 1})

	// measurement errors
	R := mat.NewDense(2, 2, []float64{100, 0, 0, 100})

	// create Kalman filter
	filter := kalman.NewFilter(X, P, F, B, Q, H, R)

	// no control
	control := mat.NewVecDense(4, nil)

	for i := 0; i < 200; i++ {
		// measure v_x and v_y with an error which is distributed according to stanard normal
		measurement := mat.NewVecDense(2, []float64{30 + rand.NormFloat64(), 10 + rand.NormFloat64()})

		// apply filter
		filtered := filter.Apply(measurement, control)

		// print out
		fmt.Fprintf(file, "%3.8f,%3.8f,%3.8f,%3.8f\n", measurement.AtVec(0), measurement.AtVec(1), filtered.AtVec(0), filtered.AtVec(1))
	}
}
