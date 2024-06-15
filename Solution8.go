package main

func getIntersectionNode__(headA, headB *ListNode) *ListNode {
	mp := make(map[*ListNode]struct{})
	for p := headA; p != nil; {
		mp[p] = struct{}{}
		p = p.Next
	}
	for p := headB; p != nil; {
		if _, ok := mp[p]; ok {
			return p
		}
		p = p.Next
	}
	return nil
}

func reverseList__(head *ListNode) *ListNode {
	var pre *ListNode
	for cur := head; cur != nil; {
		nxt := cur.Next
		cur.Next = pre
		pre, cur = cur, nxt
	}
	return pre
}

// 先找到链表的中点 反转
func isPalindrome__(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	reversedHead := reverseList(slow)
	for head != nil && reversedHead != nil {
		if head.Val != reversedHead.Val {
			return false
		}
		head = head.Next
		reversedHead = reversedHead.Next
	}
	return true
}
func hasCycle__(head *ListNode) bool {
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

func detectCycle__(head *ListNode) *ListNode {
	slow, fast := head, head
	cycle := false
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			cycle = true
			break
		}
	}
	if !cycle {
		return nil
	}
	for p := head; p != slow; {
		p = p.Next
		slow = slow.Next
	}
	return slow
}
func mergeTwoLists_(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	p1, p2, p3 := list1, list2, dummy
	for p1 != nil && p2 != nil {
		if p1.Val <= p2.Val {
			p3.Next = &ListNode{Val: p1.Val}
			p3 = p3.Next
			p1 = p1.Next
		} else {
			p3.Next = &ListNode{Val: p2.Val}
			p3 = p3.Next
			p2 = p2.Next
		}
	}
	if p1 != nil {
		p3.Next = p1
	}
	if p2 != nil {
		p3.Next = p2
	}
	return dummy.Next
}
func addTwoNumbers_(l1 *ListNode, l2 *ListNode) *ListNode {
	p1, p2, dummy := l1, l2, &ListNode{}
	p3 := dummy
	carry := 0
	for p1 != nil && p2 != nil {
		val := (carry + p1.Val + p2.Val) % 10
		carry = (carry + p1.Val + p2.Val) / 10
		p3.Next = &ListNode{Val: val}
		p1, p2, p3 = p1.Next, p2.Next, p3.Next
	}
	var p *ListNode
	if p1 != nil {
		p = p1
	} else {
		p = p2
	}
	for p != nil {
		val := (carry + p.Val) % 10
		carry = (carry + p.Val) / 10
		p3.Next = &ListNode{Val: val}
		p, p3 = p.Next, p3.Next
	}
	if carry != 0 {
		p3.Next = &ListNode{Val: 1}
	}
	return dummy.Next
}
func removeNthFromEnd_(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow, fast := dummy, dummy
	for ; n > 0; n-- {
		fast = fast.Next
	}
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next
	return dummy.Next
}
func maximumBeauty(nums []int, k int) int {

}
