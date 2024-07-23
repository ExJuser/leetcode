package common

import (
	"math/rand/v2"
	"testing"
)

var (
	slice = GenerateSlice(1000, 1000)
)

func BenchmarkReverseList(b *testing.B) {
	b.Run("迭代实现", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			list := GenerateLinkedListFromSlice(slice)
			reverseList(list)
		}
	})
	b.Run("递归实现", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			list := GenerateLinkedListFromSlice(slice)
			reverseListRecursive(list)
		}
	})
}

func BenchmarkSearchRange(b *testing.B) {
	slices := GenerateSlice(1000, 10000000)
	randNum := rand.IntN(1000)
	b.Run("两边扩散", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			searchRangePivot(slices, randNum)
		}
	})
	b.Run("二分查找", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			searchRangeBinary(slices, randNum)
		}
	})
}
