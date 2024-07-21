package dmsxl

import (
	"fmt"
	"math/rand/v2"
)

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

func GetLinkedListLength(head *ListNode) (length int) {
	for p := head; p != nil; p = p.Next {
		length++
	}
	return length
}

func Abs[T Number](num T) T {
	if num < 0 {
		return -num
	}
	return num
}

func GenerateLinkedListFromSlice(vals []int) *ListNode {
	dummy := &ListNode{}
	p := dummy
	for _, val := range vals {
		p.Next = &ListNode{Val: val}
		p = p.Next
	}
	return dummy.Next
}

func PrintLinkedList(head *ListNode) {
	vals := make([]int, 0)
	for p := head; p != nil; p = p.Next {
		vals = append(vals, p.Val)
	}
	fmt.Println(vals)
}

func SlicesSum[T Number](nums []T) (sum T) {
	for _, num := range nums {
		sum += num
	}
	return
}

func QuickSort(nums []int) []int {
	var helper func(left, right int)
	helper = func(left, right int) {
		if left >= right {
			return
		}
		//随机生成一个left到right之间的数
		pivot := nums[rand.IntN(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for nums[i] < pivot {
				i++
			}
			for nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		helper(left, j)
		helper(i, right)
	}
	helper(0, len(nums)-1)
	return nums
}

type IntMinHeap []int

func (h *IntMinHeap) Len() int {
	return len(*h)
}

func (h *IntMinHeap) Less(i, j int) bool {
	return (*h)[i] < (*h)[j]
}

func (h *IntMinHeap) Swap(i, j int) {
	(*h)[i], (*h)[j] = (*h)[j], (*h)[i]
}

func (h *IntMinHeap) Push(x any) {
	*h = append(*h, x.(int))
}

func (h *IntMinHeap) Pop() any {
	x := (*h)[h.Len()-1]
	*h = (*h)[:h.Len()-1]
	return x
}
