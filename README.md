# Kalman filtering in Golang

[![License](http://img.shields.io/badge/license-MIT-red.svg?style=flat)](https://github.com/konimarti/kalman/blob/master/LICENSE)
[![GoDoc](https://godoc.org/github.com/konimarti/observer?status.svg)](https://godoc.org/github.com/konimarti/kalman)
[![goreportcard](https://goreportcard.com/badge/github.com/konimarti/observer)](https://goreportcard.com/report/github.com/konimarti/kalman)

```go get github.com/konimarti/kalman```

## Example
```go
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

	// scaling matrix: only measure velocities
	H := mat.NewDense(2, 4, []float64{
		0, 0, 1, 0,
		0, 0, 0, 1})

	// measurement errors
	R := mat.NewDense(2, 2, []float64{100, 0, 0, 100})

	// create Kalman filter
	filter := kalman.NewFilter(X, P, F, B, H, R)

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
```

### Results

![Results of Kalman filtering on car example.](example/car/car.png)

## Credits

This software package has been developed for and is in production at [Kalkfabrik Netstal](http://www.kfn.ch/en).
