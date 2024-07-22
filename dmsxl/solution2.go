package dmsxl

import (
	"container/heap"
	"slices"
	"sort"
)

type NodeHeap []*ListNode

func (n *NodeHeap) Len() int {
	return len(*n)
}

func (n *NodeHeap) Less(i, j int) bool {
	return (*n)[i].Val < (*n)[j].Val
}

func (n *NodeHeap) Swap(i, j int) {
	(*n)[i], (*n)[j] = (*n)[j], (*n)[i]
}

func (n *NodeHeap) Push(x any) {
	*n = append(*n, x.(*ListNode))
}

func (n *NodeHeap) Pop() any {
	x := (*n)[n.Len()-1]
	*n = (*n)[:n.Len()-1]
	return x
}

// 23. 合并 K 个升序链表
func mergeKLists(lists []*ListNode) *ListNode {
	hp := &NodeHeap{}
	for _, list := range lists {
		if list != nil {
			heap.Push(hp, list)
		}
	}
	dummy := &ListNode{}
	p := dummy
	for hp.Len() != 0 {
		node := heap.Pop(hp).(*ListNode)
		p.Next = &ListNode{Val: node.Val}
		p = p.Next
		if node.Next != nil {
			heap.Push(hp, node.Next)
		}
	}
	return dummy.Next
}

// 69. x 的平方根
func mySqrt(x int) int {
	left, right := 0, x
	ans := 0
	for left <= right {
		mid := (right-left)/2 + left
		if mid*mid <= x {
			ans = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return ans
}

// 34. 在排序数组中查找元素的第一个和最后一个位置
func searchRangeBinary(nums []int, target int) []int {
	//先找到第一个
	left, right := 0, len(nums)-1
	first := -1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] >= target {
			first = mid
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	if first == -1 || nums[first] != target {
		return []int{-1, -1}
	}
	return []int{first, sort.SearchInts(nums[first:], target+1) + first - 1}
}

// 141. 环形链表
func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// 142. 环形链表 II
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			for p := head; p != slow; {
				p = p.Next
				slow = slow.Next
			}
			return slow
		}
	}
	return nil
}

// 198. 打家劫舍
func rob(nums []int) int {
	if len(nums) <= 2 {
		return slices.Max(nums)
	}
	dp1, dp2 := nums[0], max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp1, dp2 = dp2, max(dp1+nums[i], dp2)
	}
	return dp2
}
