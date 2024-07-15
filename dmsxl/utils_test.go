package dmsxl

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestGetLinkedListLength(t *testing.T) {
	linkedlist := &ListNode{Next: &ListNode{Next: &ListNode{Next: nil}}}
	assert.EqualValues(t, 3, GetLinkedListLength(linkedlist))
}
