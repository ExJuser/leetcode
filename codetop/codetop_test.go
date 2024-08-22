package main

import (
	"fmt"
	"testing"
)

func generateListFromSlice(nums []int) *ListNode {
	dummy := &ListNode{}
	cur := dummy
	for _, num := range nums {
		cur.Next = &ListNode{Val: num}
		cur = cur.Next
	}
	return dummy.Next
}

func printList(head *ListNode) {
	for cur := head; cur != nil; cur = cur.Next {
		fmt.Printf("%d ", cur.Val)
	}
	fmt.Println()
}

func TestMergeTwoLists(t *testing.T) {
	list1 := generateListFromSlice([]int{1, 2, 4, 5, 6, 8, 9})
	list2 := generateListFromSlice([]int{1, 3, 4, 5, 5, 6, 10, 12, 12, 12})
	list := mergeTwoListsDuplicate(list1, list2)
	printList(list)
}

func TestSqrt(t *testing.T) {
	fmt.Println(sqrt(586))
}
func TestConvertToTitle(t *testing.T) {
	fmt.Println(convertToTitle(701))
}
func TestReorder(t *testing.T) {
	list := generateListFromSlice([]int{1, 2, 3, 4})
	reorderList(list)
}

func TestAdd(t *testing.T) {
	num1 := "123"
	num2 := "999"
	fmt.Println(add(num1, num2))
}

func TestMultiplyByBit(t *testing.T) {
	fmt.Println(multiplyByBit("123", '9', 2))
}

func TestMultiply(t *testing.T) {
	multiply("123", "0")
}

func TestFindNumber(t *testing.T) {
	findNumberOfLIS([]int{1, 2, 4, 3, 5, 4, 7, 2})
}
