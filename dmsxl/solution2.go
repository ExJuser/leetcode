package dmsxl

import "container/heap"

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
