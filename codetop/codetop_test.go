package main

import (
	"fmt"
	"testing"
	"unsafe"
)

func Test(t *testing.T) {
	str := "123"
	fmt.Println(unsafe.Sizeof(str))
}
