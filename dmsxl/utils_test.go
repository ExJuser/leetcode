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
