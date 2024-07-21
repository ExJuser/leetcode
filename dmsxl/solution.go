package dmsxl

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// 704. 二分查找
//func search(nums []int, target int) int {
//	left, right := 0, len(nums)-1
//	for left <= right {
//		mid := (right-left)/2 + left
//		if nums[mid] == target {
//			return mid
//		} else if nums[mid] < target {
//			left = mid + 1
//		} else {
//			right = mid - 1
//		}
//	}
//	return -1
//}

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
//func reverseList(head *ListNode) *ListNode {
//	var dfs func(node *ListNode) *ListNode
//	dfs = func(node *ListNode) *ListNode {
//		if node == nil || node.Next == nil {
//			return node
//		}
//		//先将后面的节点反转
//		newHead := dfs(node.Next)
//		node.Next.Next = node
//		node.Next = nil
//		return newHead
//	}
//	return dfs(head)
//}

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
//func spiralOrder(matrix [][]int) (ans []int) {
//	height, width := len(matrix), len(matrix[0])
//	top, bottom, left, right := 0, height-1, 0, width-1
//	for left < right && top < bottom {
//		for i := left; i <= right; i++ {
//			ans = append(ans, matrix[top][i])
//		}
//		top++
//		for i := top; i <= bottom; i++ {
//			ans = append(ans, matrix[i][right])
//		}
//		right--
//		for i := right; i >= left; i-- {
//			ans = append(ans, matrix[bottom][i])
//		}
//		bottom--
//		for i := bottom; i >= top; i-- {
//			ans = append(ans, matrix[i][left])
//		}
//		left++
//	}
//	//只剩一行
//	if top == bottom {
//		for i := left; i <= right; i++ {
//			ans = append(ans, matrix[top][i])
//		}
//	} else if left == right {
//		for i := top; i <= bottom; i++ {
//			ans = append(ans, matrix[i][left])
//		}
//	}
//	return
//}

// 19. 删除链表的倒数第 N 个结点 快慢指针
func removeNthFromEnd(head *ListNode, n int) *ListNode {
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

// 142. 环形链表 II 需要输出环的入口
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			break
		}
	}
	if fast == nil || fast.Next == nil {
		return nil
	}
	for p := head; p != slow; {
		p = p.Next
		slow = slow.Next
	}
	return slow
}

// 141. 环形链表 只需要判断有环无环
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

// 242. 有效的字母异位词
func isAnagram(s string, t string) bool {
	target := make(map[byte]int)
	for _, ch := range s {
		target[byte(ch)]++
	}
	for _, ch := range t {
		target[byte(ch)]--
		if target[byte(ch)] == 0 {
			delete(target, byte(ch))
		}
	}
	return len(target) == 0
}

// 383. 赎金信
func canConstruct(ransomNote string, magazine string) bool {
	mp := make(map[byte]int)
	for _, ch := range ransomNote {
		mp[byte(ch)]++
	}
	target := len(mp)
	cur := 0
	for _, ch := range magazine {
		mp[byte(ch)]--
		if mp[byte(ch)] == 0 {
			cur++
		}
	}
	return cur == target
}

// 49. 字母异位词分组 排序即可
func groupAnagrams(strs []string) [][]string {
	cnt := 0
	mp := make(map[string]int)
	ans := make([][]string, 0)
	for _, str := range strs {
		bytes := []byte(str)
		slices.Sort(bytes)
		sortedStr := string(bytes)
		if index, ok := mp[sortedStr]; ok {
			ans[index] = append(ans[index], str)
		} else {
			mp[sortedStr] = cnt
			ans = append(ans, make([]string, 0))
			ans[cnt] = append(ans[cnt], str)
			cnt++
		}
	}
	return ans
}

// 438. 找到字符串中所有字母异位词
func findAnagrams(s string, p string) (ans []int) {
	if len(s) < len(p) {
		return
	}
	target := make(map[byte]int)
	for _, ch := range p {
		target[byte(ch)]++
	}
	for i := 0; i < len(p); i++ {
		target[s[i]]--
		if target[s[i]] == 0 {
			delete(target, s[i])
		}
	}
	if len(target) == 0 {
		ans = append(ans, 0)
	}
	for i := 1; i < len(s)-len(p)+1; i++ {
		target[s[i-1]]++
		if target[s[i-1]] == 0 {
			delete(target, s[i-1])
		}
		//abcd abc
		target[s[i+len(p)-1]]--
		if target[s[i+len(p)-1]] == 0 {
			delete(target, s[i+len(p)-1])
		}
		if len(target) == 0 {
			ans = append(ans, i)
		}
	}
	return
}

// 349. 两个数组的交集
func intersection(nums1 []int, nums2 []int) (ans []int) {
	mp := make(map[int]struct{}, len(nums1))
	for _, num := range nums1 {
		mp[num] = struct{}{}
	}
	for _, num := range nums2 {
		if _, ok := mp[num]; ok {
			ans = append(ans, num)
			delete(mp, num) //加入交集即可删除
		}
	}
	return
}

// 350. 两个数组的交集 II
func intersect(nums1 []int, nums2 []int) (ans []int) {
	set1 := make(map[int]int)
	set2 := make(map[int]int)
	for _, num := range nums1 {
		set1[num]++
	}
	for _, num := range nums2 {
		set2[num]++
	}
	for k, v1 := range set1 {
		if v2, ok := set2[k]; ok && v2 > 0 {
			for i := 0; i < min(v1, v2); i++ {
				ans = append(ans, k)
			}
			set2[k] -= min(v1, v2)
		}
	}
	return
}

// 202. 快乐数
func isHappy(n int) bool {
	set := make(map[int]struct{})
	for n != 1 {
		sum := 0
		for n > 0 {
			digit := n % 10
			sum += digit * digit
			n /= 10
		}
		if _, ok := set[sum]; ok {
			return false
		}
		set[sum] = struct{}{}
		n = sum
	}
	return true
}

// 1. 两数之和
func twoSum(nums []int, target int) []int {
	mp := make(map[int]int)
	for i, num := range nums {
		if j, ok := mp[target-num]; ok {
			return []int{i, j}
		}
		mp[num] = i
	}
	return []int{}
}

// 2956. 找到两个数组中的公共元素
func findIntersectionValues(nums1 []int, nums2 []int) []int {
	set1 := make(map[int]struct{})
	set2 := make(map[int]struct{})
	var ans1, ans2 int
	for _, num := range nums1 {
		set1[num] = struct{}{}
	}
	for _, num := range nums2 {
		set2[num] = struct{}{}
		if _, ok := set1[num]; ok {
			ans2++
		}
	}
	for _, num := range nums1 {
		if _, ok := set2[num]; ok {
			ans1++
		}
	}
	return []int{ans1, ans2}
}

func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) (ans int) {
	mp := make(map[int]int)
	for _, num1 := range nums1 {
		for _, num2 := range nums2 {
			mp[num1+num2]++
		}
	}
	for _, num3 := range nums3 {
		for _, num4 := range nums4 {
			ans += mp[-num3-num4]
		}
	}
	return
}

func spiralOrder(matrix [][]int) (ans []int) {
	left, right := 0, len(matrix[0])-1
	top, bottom := 0, len(matrix)-1
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
	if left == right {
		for i := top; i <= bottom; i++ {
			ans = append(ans, matrix[i][left])
		}
	} else if top == bottom {
		for i := left; i <= right; i++ {
			ans = append(ans, matrix[top][i])
		}
	}
	return
}

// 20. 有效的括号
func isValid(s string) bool {
	mp := map[byte]byte{
		')': '(',
		']': '[',
		'}': '{',
	}
	stack := make([]byte, 0, len(s))
	for _, ch := range s {
		if _, ok := mp[byte(ch)]; !ok { //是左括号
			stack = append(stack, byte(ch))
		} else { //右括号
			if len(stack) == 0 || mp[byte(ch)] != stack[len(stack)-1] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

// 46. 全排列
func permute(nums []int) (ans [][]int) {
	used := make([]bool, len(nums))
	var dfs func(index int, path []int)
	dfs = func(index int, path []int) {
		if index == len(nums) {
			ans = append(ans, append([]int{}, path...))
			return
		}
		for i := 0; i < len(nums); i++ {
			if !used[i] {
				used[i] = true
				path = append(path, nums[i])
				dfs(index+1, path)
				used[i] = false
				path = path[:len(path)-1]
			}
		}
	}
	dfs(0, []int{})
	return
}

// 3. 无重复字符的最长子串 滑动窗口
func lengthOfLongestSubstring(s string) int {
	mp := make(map[byte]int)
	var left, ans int
	for right := 0; right < len(s); right++ {
		mp[s[right]]++
		for ; len(mp) < right-left+1; left++ {
			mp[s[left]]--
			if mp[s[left]] == 0 {
				delete(mp, s[left])
			}
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

// 5. 最长回文子串 动态规划
func longestPalindrome(s string) string {
	dp := make([][]int, len(s))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(s))
	}
	//dp[i][j]可以由dp[i+1][j-1]推出
	//初始对角线：单个字符的字符串都是回文串
	for i := 0; i < len(s); i++ {
		dp[i][i] = 1
	}
	length := 1
	ans := s[0:1]
	//注意遍历的顺序问题
	for j := 1; j < len(s); j++ {
		for i := 0; i < j; i++ {
			if s[i] == s[j] && dp[i+1][j-1] == j-i-1 {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = 0
			}
			if dp[i][j] > length {
				length = dp[i][j]
				ans = s[i : j+1]
			}
		}
	}
	return ans
}

// 93. 复原 IP 地址 回溯插入三个点/找到三个切割位置
func restoreIpAddresses(s string) (ans []string) {
	//如何判断是否合法：切割出来的数字处于0-255 不能有前导0
	var dfs func(pre, cur int, path []string)
	var check func(str string) bool
	check = func(str string) bool {
		//如果首位是0 那他必须是零本身
		if str[0] == '0' {
			return len(str) == 1
		}
		val, _ := strconv.Atoi(str)
		return val >= 0 && val <= 255
	}
	dfs = func(pre, cur int, path []string) {
		if cur == len(s) {
			if len(path) == 4 && cur == pre {
				ans = append(ans, strings.Join(path, "."))
			}
			return
		}
		//切割
		//判断是否合法
		if check(s[pre : cur+1]) {
			path = append(path, s[pre:cur+1])
			dfs(cur+1, cur+1, path)
			path = path[:len(path)-1]
		}
		//不切割
		dfs(pre, cur+1, path)
	}
	dfs(0, 0, []string{})
	return
}

// 415. 字符串相加
func addStrings(num1 string, num2 string) string {
	carry := 0
	bytes1 := []byte(num1)
	bytes2 := []byte(num2)
	res := make([]byte, max(len(num1), len(num2)))
	index := len(res)
	i, j := len(bytes1)-1, len(bytes2)-1
	for ; i >= 0 && j >= 0; i, j = i-1, j-1 {
		val := carry + int(bytes1[i]-'0') + int(bytes2[j]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	for ; i >= 0; i-- {
		val := carry + int(bytes1[i]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	for ; j >= 0; j-- {
		val := carry + int(bytes2[j]-'0')
		carry = val / 10
		val %= 10
		res[index] = byte(val) + '0'
		index--
	}
	if carry != 0 {
		return fmt.Sprintf("%d%s", carry, string(res))
	}
	return string(res)
}

// LCR 181. 字符串中的单词反转 使用库函数版本
//func reverseMessage(message string) string {
//	fields := strings.Fields(message)
//	slices.Reverse(fields)
//	return strings.Join(fields, " ")
//}

// LCR 181. 字符串中的单词反转 不使用库函数版本
func reverseMessage(message string) string {
	//从不是空格的字符开始循环 到是空格的循环为止
	fields := make([]string, 0, len(message))
	for left := 0; left < len(message); {
		if message[left] == ' ' {
			left++
			continue
		}
		right := left + 1
		for ; right < len(message); right++ {
			if message[right] == ' ' {
				break
			}
		}
		fields = append(fields, message[left:right])
		left = right + 1
	}
	slices.Reverse(fields)
	return strings.Join(fields, " ")
}

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

// 21. 合并两个有序链表
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := &ListNode{}
	p := dummy
	p1, p2 := list1, list2
	for p1 != nil && p2 != nil {
		if p1.Val <= p2.Val {
			p.Next = &ListNode{Val: p1.Val}
			p1 = p1.Next
		} else {
			p.Next = &ListNode{Val: p2.Val}
			p2 = p2.Next
		}
		p = p.Next
	}
	if p1 != nil {
		p.Next = p1
	} else {
		p.Next = p2
	}
	return dummy.Next
}

// 206. 反转链表的迭代做法
//
//	func reverseList(head *ListNode) *ListNode {
//		var pre *ListNode
//		for cur := head; cur != nil; {
//			nxt := cur.Next
//			cur.Next = pre
//			cur, pre = nxt, cur
//		}
//		return pre
//	}
//
// 206. 反转链表的递归
func reverseList(head *ListNode) *ListNode {
	var dfs func(node *ListNode) *ListNode
	dfs = func(node *ListNode) *ListNode {
		if node == nil || node.Next == nil {
			return node
		}
		newHead := dfs(node.Next)
		node.Next.Next = node
		node.Next = nil
		return newHead
	}
	return dfs(head)
}
