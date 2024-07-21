package dmsxl

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestGetLinkedListLength(t *testing.T) {
	linkedlist := &ListNode{Next: &ListNode{Next: &ListNode{Next: nil}}}
	assert.EqualValues(t, 3, GetLinkedListLength(linkedlist))
}

func TestGenerateLinkedListFromSlice(t *testing.T) {
	vals := []int{1, 2, 3, 4, 5}
	list := GenerateLinkedListFromSlice(vals)
	assert.EqualValues(t, 5, GetLinkedListLength(list))
}

func TestPrintLinkedList(t *testing.T) {
	PrintLinkedList(GenerateLinkedListFromSlice([]int{1, 2, 3, 4, 5, 6}))
}

func TestSlicesSum(t *testing.T) {
	nums1 := []int{1, 2, 3, 4, 5}
	nums2 := []float64{1.0, 2.1, 3.2, 4.3, 5.4}
	sum1 := SlicesSum(nums1)
	sum2 := SlicesSum(nums2)
	assert.EqualValues(t, 15, sum1)
	assert.EqualValues(t, 16.0, sum2)
}
