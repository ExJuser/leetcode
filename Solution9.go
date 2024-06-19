package main

import (
	"fmt"
	"math"
	"math/rand"
)

func maximalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
	}
	ans := 0
	for i := 0; i < len(dp); i++ {
		for j := 0; j < len(dp[i]); j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = 1
				ans = max(ans, dp[i][j])
			}
		}
	}
	for i := 1; i < len(dp); i++ {
		for j := 1; j < len(dp[i]); j++ {
			if dp[i][j] == 1 {
				x := min(int(math.Sqrt(float64(dp[i-1][j]))), int(math.Sqrt(float64(dp[i-1][j-1]))), int(math.Sqrt(float64(dp[i][j-1])))) + 1
				dp[i][j] = x * x
				ans = max(ans, dp[i][j])
			}
		}
	}
	return ans
}
func countSquares(matrix [][]int) (ans int) {
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			if i != 0 && j != 0 && matrix[i][j] == 1 {
				matrix[i][j] = min(matrix[i-1][j], matrix[i-1][j-1], matrix[i][j-1]) + 1
			}
			ans += matrix[i][j]
		}
	}
	return
}

// 快速排序
func quickSort(nums []int) {
	var helper func(int, int)
	helper = func(left, right int) {
		if left >= right {
			return
		}
		pivot := nums[rand.Intn(right-left+1)+left]
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
}

//	func quickSort_(nums []int) {
//		var helper func(left, right int)
//		helper = func(left, right int) {
//			if left >= right {
//				return
//			}
//			pivot := nums[rand.Intn(right-left+1)+left]
//			i, j := left, right
//			for i <= j {
//				for nums[i] < pivot {
//					i++
//				}
//				for nums[j] > pivot {
//					j--
//				}
//				if i <= j {
//					nums[i], nums[j] = nums[j], nums[i]
//					i++
//					j--
//				}
//			}
//			helper(left, j)
//			helper(i, right)
//		}
//	}
func findKthLargest(nums []int, k int) int {
	var helper func(left, right, k int) int
	helper = func(left, right, k int) int {
		if left >= right {
			return nums[k]
		}
		pivot := nums[rand.Intn(right-left+1)+left]
		i, j := left, right
		for i <= j {
			for i <= j && nums[i] < pivot {
				i++
			}
			for i <= j && nums[j] > pivot {
				j--
			}
			if i <= j {
				nums[i], nums[j] = nums[j], nums[i]
				i++
				j--
			}
		}
		if j >= k {
			return helper(left, j, k)
		} else {
			return helper(i, right, k)
		}
	}
	return helper(0, len(nums)-1, len(nums)-k)
}

// 冒泡排序：每一轮确定一个最大值冒泡冒到最后一个为止
func bubbleSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}

// 选择排序：每一轮确定一个最小值与对应位置元素交换
func selectSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		minIndex := i
		for j := i + 1; j < len(nums); j++ {
			if nums[j] < nums[minIndex] {
				minIndex = j
			}
		}
		nums[i], nums[minIndex] = nums[minIndex], nums[i]
	}
}

// 其实就是选出第len(nums)/2大的数
func majorityElement(nums []int) int {
	return findKthLargest(nums, len(nums)/2+1)
}

// 前缀积和后缀积
func productExceptSelf(nums []int) []int {
	prefix := make([]int, len(nums)+1)
	suffix := make([]int, len(nums)+1)
	prefix[0] = 1
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] * nums[i]
	}
	suffix[len(nums)] = 1
	for i := len(nums) - 1; i >= 0; i-- {
		suffix[i] = suffix[i+1] * nums[i]
	}
	fmt.Println(prefix)
	fmt.Println(suffix)
	for i := 0; i < len(nums); i++ {
		nums[i] = prefix[i] * suffix[i+1]
	}
	return nums
}

// MinStack 双栈解法
//type MinStack struct {
//	stack, minStack []int
//}
//
//func Constructor() MinStack {
//	return MinStack{
//		stack:    make([]int, 0, 1000),
//		minStack: make([]int, 0, 1000),
//	}
//}
//
//func (this *MinStack) Push(val int) {
//	this.stack = append(this.stack, val)
//	if len(this.minStack) == 0 || val < this.minStack[len(this.minStack)-1] {
//		this.minStack = append(this.minStack, val)
//	} else {
//		this.minStack = append(this.minStack, this.minStack[len(this.minStack)-1])
//	}
//}
//
//func (this *MinStack) Pop() {
//	this.stack = this.stack[:len(this.stack)-1]
//	this.minStack = this.minStack[:len(this.minStack)-1]
//}
//
//func (this *MinStack) Top() int {
//	return this.stack[len(this.stack)-1]
//}
//
//func (this *MinStack) GetMin() int {
//	return this.minStack[len(this.minStack)-1]
//}

func maxProduct(nums []int) int {
	minDP := make([]float64, len(nums))
	maxDP := make([]float64, len(nums))
	minDP[0], maxDP[0] = float64(nums[0]), float64(nums[0])
	ans := float64(nums[0])
	for i := 1; i < len(nums); i++ {
		minDP[i] = min(minDP[i-1]*float64(nums[i]), maxDP[i-1]*float64(nums[i]), float64(nums[i]))
		maxDP[i] = max(minDP[i-1]*float64(nums[i]), maxDP[i-1]*float64(nums[i]), float64(nums[i]))
		ans = max(ans, maxDP[i], minDP[i])
	}
	return int(ans)
}

//	func partitionNode(head *ListNode, x int) *ListNode {
//		leftDummy := &ListNode{}
//		rightDummy := &ListNode{}
//		p, q := leftDummy, rightDummy
//		for cur := head; cur != nil; cur = cur.Next {
//			if cur.Val < x {
//				p.Next = &ListNode{Val: cur.Val}
//				p = p.Next
//			} else {
//				q.Next = &ListNode{Val: cur.Val}
//				q = q.Next
//			}
//		}
//		p.Next = rightDummy.Next
//		return leftDummy.Next
//	}
func partitionNode(head *ListNode, x int) *ListNode {
	dummy := &ListNode{Next: head}
	slow := dummy
	for slow != nil && slow.Next != nil && slow.Next.Val < x {
		slow = slow.Next
	}
	fast := slow
	for fast != nil && fast.Next != nil {
		if fast.Next.Val < x {
			temp := fast.Next
			fast.Next = temp.Next
			temp.Next = slow.Next
			slow.Next = temp
			slow = slow.Next
		} else {
			fast = fast.Next
		}
	}
	return dummy.Next
}

// 自顶向下的归并排序
//func sortList(head *ListNode) *ListNode {
//	var dfs func(node *ListNode) *ListNode
//	dfs = func(node *ListNode) *ListNode {
//		if node == nil || node.Next == nil {
//			return node
//		}
//		slow, fast := node, node
//		var preSlow *ListNode
//		for fast != nil && fast.Next != nil {
//			preSlow = slow
//			slow = slow.Next
//			fast = fast.Next.Next
//		}
//		preSlow.Next = nil
//		left, right := dfs(node), dfs(slow)
//		dummy := &ListNode{}
//		p := dummy
//		for left != nil && right != nil {
//			if left.Val <= right.Val {
//				p.Next = &ListNode{Val: left.Val}
//				left = left.Next
//			} else {
//				p.Next = &ListNode{Val: right.Val}
//				right = right.Next
//			}
//			p = p.Next
//		}
//		if left != nil {
//			p.Next = left
//		} else {
//			p.Next = right
//		}
//		return dummy.Next
//	}
//	return dfs(head)
//}

func merge(head1, head2 *ListNode) *ListNode {
	dummyHead := &ListNode{}
	temp, temp1, temp2 := dummyHead, head1, head2
	for temp1 != nil && temp2 != nil {
		if temp1.Val <= temp2.Val {
			temp.Next = temp1
			temp1 = temp1.Next
		} else {
			temp.Next = temp2
			temp2 = temp2.Next
		}
		temp = temp.Next
	}
	if temp1 != nil {
		temp.Next = temp1
	} else if temp2 != nil {
		temp.Next = temp2
	}
	return dummyHead.Next
}

func sortList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}
	length := 0
	for node := head; node != nil; node = node.Next {
		length++
	}
	dummyHead := &ListNode{Next: head}
	for subLength := 1; subLength < length; subLength <<= 1 {
		prev, cur := dummyHead, dummyHead.Next
		for cur != nil {
			head1 := cur
			for i := 1; i < subLength && cur.Next != nil; i++ {
				cur = cur.Next
			}
			head2 := cur.Next
			cur.Next = nil
			cur = head2
			for i := 1; i < subLength && cur != nil && cur.Next != nil; i++ {
				cur = cur.Next
			}
			var next *ListNode
			if cur != nil {
				next = cur.Next
				cur.Next = nil
			}
			prev.Next = merge(head1, head2)
			for prev.Next != nil {
				prev = prev.Next
			}
			cur = next
		}
	}
	return dummyHead.Next
}

// 链表的插入排序
func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	pre := dummy
	for cur := head; cur != nil; {
		p := dummy
		for p.Next.Val <= cur.Val && p.Next != cur {
			p = p.Next
		}
		if p.Next != cur {
			temp := cur.Next
			pre.Next = temp
			cur.Next = p.Next
			p.Next = cur
			cur = temp
		} else {
			pre = cur
			cur = cur.Next
		}
	}
	return dummy.Next
}
