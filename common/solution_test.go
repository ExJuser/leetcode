package common

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"math/rand/v2"
	"slices"
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

func TestSort(t *testing.T) {
	nums := GenerateSlice(1000, 100000)
	numsCopy := make([]int, len(nums))
	copy(numsCopy, nums)
	quickSort(nums)
	//selectSort(numsCopy)
	bubbleSort(numsCopy)
	assert.True(t, slices.Equal(nums, numsCopy))
}

func TestDuplicate(t *testing.T) {
	list := GenerateLinkedListFromSlice([]int{1, 2, 3, 3, 4, 4, 5})
	PrintLinkedList(list)
	newList := deleteDuplicates(list)
	PrintLinkedList(newList)
}

func TestQuickSort(t *testing.T) {
	nums := GenerateSlice(1000, 10000)
	numsCopy := make([]int, len(nums))
	copy(numsCopy, nums)
	fmt.Println(nums)
	fmt.Println(numsCopy)
	quickSort_(nums)
	slices.Sort(numsCopy)
	assert.Equal(t, nums, numsCopy)
	fmt.Println(nums)
	fmt.Println(numsCopy)
}

func BenchmarkSort(b *testing.B) {
	nums := GenerateSlice(10, 10000)
	numsCopy := make([]int, len(nums))
	copy(numsCopy, nums)
	b.Run("builtin sort", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			slices.Sort(nums)
		}
	})
	b.Run("my quick sort", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			quickSort_(numsCopy)
		}
	})
}
