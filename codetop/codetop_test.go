package main

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	fmt.Println(mergeSort([]int{1, 5, 5, 89, 4, 2, 4, 55, 6, 85, 2}))
}
