package hot100

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPrefixAndSuffix(t *testing.T) {
	testCase := []struct {
		input  []int
		output []int
	}{
		{
			input:  []int{1, 2, 3, 4},
			output: []int{24, 12, 8, 6},
		},
		{
			input:  []int{-1, -1, 0, -3, -3},
			output: []int{0, 0, 9, 0, 0},
		},
	}
	for _, tc := range testCase {
		assert.Equal(t, tc.output, productExceptSelf(tc.input))
	}
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

func PrintLinkedList(node *ListNode) {
	for p := node; p != nil; p = p.Next {
		fmt.Print(p.Val)
	}
	fmt.Println()
}

func TestMergeList(t *testing.T) {
	list1 := GenerateLinkedListFromSlice([]int{1, 2, 3})
	list2 := GenerateLinkedListFromSlice([]int{2, 3, 4})
	PrintLinkedList(mergeList(list1, list2))
}

func TestWordBreak(t *testing.T) {
	fmt.Println(wordBreak("catsandog", []string{"cats", "dog", "sand", "and", "cat"}))
}

func TestSingleNumber(t *testing.T) {
	nums := []int{1, 1, 2, 1, 1, 2, 2, 6}
	temp := nums[0]
	for i := 1; i < len(nums); i++ {
		temp ^= nums[i]
	}
	fmt.Println(temp)
}
