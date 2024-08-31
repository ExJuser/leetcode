package main

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	nums := []int{59, 9, 5, 5, 2, 4, 41, 56, 45, 6, 41, 565}
	fmt.Println(bubbleSort(nums))
}
