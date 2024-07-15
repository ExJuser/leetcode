package dmsxl

import (
	"math"
)

// 704. 二分查找
func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

// 35. 搜索插入位置
func searchInsert(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := len(nums)
	for left <= right {
		mid := (right-left)/2 + left
		if nums[mid] == target {
			return mid
		} else if nums[mid] < target {
			left = mid + 1
		} else {
			ans = mid
			right = mid - 1
		}
	}
	return ans
}

// 34. 在排序数组中查找元素的第一个和最后一个位置
func searchRange(nums []int, target int) []int {
	pivot := search(nums, target)
	if pivot == -1 {
		return []int{-1, -1}
	}
	left, right := pivot, pivot
	for left >= 0 && nums[left] == target {
		left--
	}
	for right < len(nums) && nums[right] == target {
		right++
	}
	return []int{left + 1, right - 1}
}

// 69. x 的平方根
func mySqrt(x int) int {
	left, right := 0, x
	var ans int
	for left <= right {
		mid := (right-left)/2 + left
		if mid*mid > x {
			right = mid - 1
		} else {
			ans = mid
			left = mid + 1
		}
	}
	return ans
}

// 367. 有效的完全平方数
func isPerfectSquare(num int) bool {
	left, right := 1, num
	for left <= right {
		mid := (right-left)/2 + left
		square := mid * mid
		if square == num {
			return true
		} else if square < num {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return false
}

// 27. 移除元素
func removeElement(nums []int, val int) int {
	end := len(nums) - 1
	for i := 0; i <= end; {
		if nums[i] == val {
			nums[i], nums[end] = nums[end], nums[i]
			end--
		} else {
			i++
		}
	}
	return end + 1
}

// 26. 删除有序数组中的重复项
func removeDuplicates(nums []int) int {
	var count int
	for i := 0; i < len(nums); i++ {
		if i == 0 || nums[i] != nums[i-1] {
			nums[count] = nums[i]
			count++
		}
	}
	return count
}

// 283. 移动零
func moveZeroes(nums []int) {
	index := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		}
	}
	nums = append(nums[:index], make([]int, len(nums)-index)...)
}

// 844. 比较含退格的字符串
func backspaceCompare(s string, t string) bool {
	var help func(str string) string
	help = func(str string) string {
		bytes := []byte(str)
		index := 0
		for i := 0; i < len(bytes); i++ {
			if bytes[i] == '#' {
				index = max(index-1, 0)
			} else {
				bytes[index] = bytes[i]
				index++
			}
		}
		return string(bytes[:index])
	}
	return help(s) == help(t)
}

// 977. 有序数组的平方
func sortedSquares(nums []int) []int {
	newNums := make([]int, len(nums))
	insLoc := len(nums) - 1
	left, right := 0, len(nums)-1
	for left <= right {
		if nums[left]*nums[left] <= nums[right]*nums[right] {
			newNums[insLoc] = nums[right] * nums[right]
			right--
		} else {
			newNums[insLoc] = nums[left] * nums[left]
			left++
		}
		insLoc--
	}
	return newNums
}

// 209. 长度最小的子数组
func minSubArrayLen(target int, nums []int) int {
	var left, sum int
	ans := math.MaxInt
	for right := 0; right < len(nums); right++ {
		sum += nums[right]
		for ; sum >= target; left++ {
			sum -= nums[left]
			ans = min(ans, right-left+1)
		}
	}
	if ans == math.MaxInt {
		return 0
	}
	return ans
}

// 904. 水果成篮
func totalFruit(fruits []int) int {
	mp := make(map[int]int)
	var left, ans int
	for right := 0; right < len(fruits); right++ {
		mp[fruits[right]]++
		for ; len(mp) > 2; left++ {
			mp[fruits[left]]--
			if mp[fruits[left]] == 0 {
				delete(mp, fruits[left])
			}
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

// 76. 最小覆盖子串
//func minWindow(s string, t string) string {
//	var check func(mp map[byte]int) bool
//	check = func(mp map[byte]int) bool {
//		for _, v := range mp {
//			if v > 0 {
//				return false
//			}
//		}
//		return true
//	}
//	mp := make(map[byte]int)
//	left, length := 0, len(s)+1
//	ans := ""
//	for _, ch := range t {
//		mp[byte(ch)]++
//	}
//	for right := 0; right < len(s); right++ {
//		mp[s[right]]--
//		for ; check(mp); left++ {
//			if right-left+1 < length {
//				length = right - left + 1
//				ans = s[left : right+1]
//			}
//			mp[s[left]]++
//		}
//	}
//	return ans
//}

// 76. 最小覆盖子串的优化版本 额外使用一个变量维护当前已经满足要求的字符个数
func minWindow(s string, t string) string {
	var cur, left int
	length, ans := len(s)+1, ""
	target := make(map[byte]int)
	exist := make(map[byte]struct{})
	for _, ch := range t {
		target[byte(ch)]++
		exist[byte(ch)] = struct{}{}
	}
	for right := 0; right < len(s); right++ {
		target[s[right]]--
		if _, ok := exist[s[right]]; ok && target[s[right]] == 0 {
			cur++
		}
		for ; cur == len(exist); left++ {
			target[s[left]]++
			if length > right-left+1 {
				length = right - left + 1
				ans = s[left : right+1]
			}
			if _, ok := exist[s[left]]; ok && target[s[left]] == 1 {
				cur--
			}
		}
	}
	return ans
}

// 203. 移除链表元素
func removeElements(head *ListNode, val int) *ListNode {
	dummy := &ListNode{Next: head}
	for p := dummy; p != nil && p.Next != nil; {
		if p.Next.Val == val {
			p.Next = p.Next.Next
		} else {
			p = p.Next
		}
	}
	return dummy.Next
}

// 206. 反转链表的迭代法
//func reverseList(head *ListNode) *ListNode {
//	var pre *ListNode
//	for cur := head; cur != nil; {
//		nxt := cur.Next
//		cur.Next = pre
//		cur, pre = nxt, cur
//	}
//	return pre
//}

// 206. 反转链表的递归法
func reverseList(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		//先将后面的节点反转
		newHead := dfs(node.Next)
		node.Next.Next = node
		node.Next = nil
		return newHead
	}
	return dfs(head)
}

// 24. 两两交换链表中的节点 多用几个变量表示节点之间的关系即可
//func swapPairs(head *ListNode) *ListNode {
//	dummy := &ListNode{Next: head}
//	pre := dummy
//	cur := dummy.Next
//	for cur != nil && cur.Next != nil {
//		nxt := cur.Next
//		cur.Next = nxt.Next
//		nxt.Next = cur
//		pre.Next = nxt
//		pre = cur
//		cur = cur.Next
//	}
//	return dummy.Next
//}

// 24. 两两交换链表中的节点优化版
//func swapPairs(head *ListNode) *ListNode {
//	dummy := &ListNode{Next: head}
//	for pre, cur := dummy, dummy.Next; cur != nil && cur.Next != nil; pre, cur = cur, cur.Next {
//		nxt := cur.Next
//		cur.Next = nxt.Next
//		nxt.Next = cur
//		pre.Next = nxt
//	}
//	return dummy.Next
//}

// 24. 两两交换链表中的节点递归版 画个图就可以
func swapPairs(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		nxt := node.Next
		node.Next = dfs(node.Next.Next)
		nxt.Next = node
		return nxt
	}
	return dfs(head)
}

// 59. 螺旋矩阵 II
func generateMatrix(n int) [][]int {
	top, bottom, left, right := 0, n-1, 0, n-1
	num := 1
	matrix := make([][]int, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]int, n)
	}
	for num <= n*n {
		for i := left; i <= right; i++ {
			matrix[top][i] = num
			num++
		}
		top++
		for i := top; i <= bottom; i++ {
			matrix[i][right] = num
			num++
		}
		right--
		for i := right; i >= left; i-- {
			matrix[bottom][i] = num
			num++
		}
		bottom--
		for i := bottom; i >= top; i-- {
			matrix[i][left] = num
			num++
		}
		left++
	}
	return matrix
}

// 54. 螺旋矩阵 注意处理最后只剩一行或者只剩一圈的问题
func spiralOrder(matrix [][]int) (ans []int) {
	height, width := len(matrix), len(matrix[0])
	top, bottom, left, right := 0, height-1, 0, width-1
	for left < right && top < bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
		top++
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][right])
		}
		right--
		for i := right; i >= left; i-- {
			ans = append(ans, matrix[bottom][i])
		}
		bottom--
		for i := bottom; i >= top; i-- {
			ans = append(ans, matrix[i][left])
		}
		left++
	}
	//只剩一行
	if top == bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
	} else if left == right {
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][left])
		}
	}
	return
}

// 19. 删除链表的倒数第 N 个结点 快慢指针
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	slow, fast := dummy, dummy
	for i := 0; i < n; i++ {
		fast = fast.Next
	}
	for fast.Next != nil {
		slow = slow.Next
		fast = fast.Next
	}
	slow.Next = slow.Next.Next

	return dummy.Next
}

// 面试题 02.07. 链表相交 使用额外内存的解法
//func getIntersectionNode(headA, headB *ListNode) *ListNode {
//	set := make(map[*ListNode]struct{})
//	for p := headA; p != nil; p = p.Next {
//		set[p] = struct{}{}
//	}
//	for p := headB; p != nil; p = p.Next {
//		if _, ok := set[p]; ok {
//			return p
//		}
//	}
//	return nil
//}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	lenA, lenB := GetLinkedListLength(headA), GetLinkedListLength(headB)
	var p, q *ListNode
	diff := lenA - lenB
	if diff > 0 {
		p = headA
		q = headB
	} else {
		p = headB
		q = headA
	}
	for i := 0; i < Abs(diff); i++ {
		p = p.Next
	}
	for p != nil && q != nil {
		if p == q {
			return p
		}
		p = p.Next
		q = q.Next
	}
	return nil
}
