package dmsxl

import (
	"fmt"
)

func GetLinkedListLength(head *ListNode) (length int) {
	for p := head; p != nil; p = p.Next {
		length++
	}
	return length
}

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
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
