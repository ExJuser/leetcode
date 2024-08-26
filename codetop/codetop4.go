package main

import (
	"math/rand/v2"
)

// 5. 最长回文子串
func longestPalindrome(s string) string {
	n := len(s)
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	ans := ""
	for i := n - 1; i >= 0; i-- {
		for j := i; j < n; j++ {
			if s[i] == s[j] {
				if i == j || j-i == 1 {
					dp[i][j] = true
				} else {
					dp[i][j] = dp[i+1][j-1]
				}
				if dp[i][j] && j-i+1 > len(ans) {
					ans = s[i : j+1]
				}
			}
		}
	}
	return ans
}

// 1143. 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	//dpij 以i-1 j-1为结尾的最长公共子序列 这样创建dp数组不需要额外的初始化
	dp := make([][]int, len(text1)+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(text2)+1)
	}
	for i := 0; i < len(text1); i++ {
		for j := 0; j < len(text2); j++ {
			if text1[i] == text2[j] {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
			}
		}
	}
	return dp[len(text1)][len(text2)]
}

// 763. 划分字母区间
func partitionLabels(s string) (ans []int) {
	//统计每一个字符出现的最晚位置 遇到一个字符就向右扩充到最晚位置
	lastAppear := make(map[byte]int)
	for loc, ch := range s {
		lastAppear[byte(ch)] = loc
	}
	var maxRight, j int
	for i := 0; i < len(s); i++ {
		maxRight = max(maxRight, lastAppear[s[i]])
		j++
		if i == maxRight { //完成一个片段的收集
			ans = append(ans, j)
			j = 0
		}
	}
	return
}

// 39. 组合总和
func combinationSum(candidates []int, target int) (ans [][]int) {
	var dfs func(index, sum int, path []int)
	dfs = func(index, sum int, path []int) {
		if sum > target {
			return
		}
		if index == len(candidates) {
			if sum == target {
				ans = append(ans, append([]int{}, path...))
			}
			return
		}
		path = append(path, candidates[index])
		dfs(index, sum+candidates[index], path)
		path = path[:len(path)-1]

		dfs(index+1, sum, path)
	}
	dfs(0, 0, []int{})
	return
}

func merge2(nums1 []int, m int, nums2 []int, n int) {
	i, j := m-1, n-1
	for index := m + n - 1; index >= 0; index-- {
		if i >= 0 && j >= 0 {
			if nums1[i] >= nums2[j] {
				nums1[index] = nums1[i]
				i--
			} else {
				nums1[index] = nums2[j]
				j--
			}
		} else if i >= 0 {
			nums1[index] = nums1[i]
			i--
		} else {
			nums1[index] = nums2[j]
			j--
		}
	}
}

type Solution struct {
	nodes []*ListNode
}

//func Constructor(head *ListNode) Solution {
//	nodes := make([]*ListNode, 0)
//	for cur := head; cur != nil; cur = cur.Next {
//		nodes = append(nodes, cur)
//	}
//	return Solution{nodes: nodes}
//}

func (this *Solution) GetRandom() int {
	return this.nodes[rand.IntN(len(this.nodes))].Val
}

// 547. 省份数量
func findCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	father := make([]int, n)
	for i := 0; i < len(father); i++ {
		father[i] = i
	}
	var merge func(i, j int) bool
	var find func(i int) int
	find = func(i int) int {
		if father[i] != i {
			father[i] = find(father[i])
		}
		return father[i]
	}
	merge = func(x, y int) bool {
		fx, fy := find(x), find(y)
		if fx != fy {
			father[fx] = fy
			return true
		}
		return false
	}
	var ans = n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if isConnected[i][j] == 1 {
				if merge(i, j) {
					ans--
				}
			}
		}
	}
	return ans
}

// 234. 回文链表
func isPalindrome(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	var reverse func(node *ListNode) *ListNode
	reverse = func(node *ListNode) *ListNode {
		var pre *ListNode
		for cur := node; cur != nil; {
			nxt := cur.Next
			cur.Next = pre
			pre, cur = cur, nxt
		}
		return pre
	}
	for p, q := head, reverse(slow); p != nil && q != nil; p, q = p.Next, q.Next {
		if p.Val != q.Val {
			return false
		}
	}
	return true
}

func findTargetSumWays(nums []int, target int) int {
	var sum int
	for _, num := range nums {
		sum += num
	}
	diff := sum - target
	if diff%2 != 0 {
		return 0
	}
	bag := diff / 2
	dp := make([][]int, len(nums))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, bag+1)
	}
	dp[0][0] += 1
	if nums[0] <= bag {
		dp[0][nums[0]] += 1
	}
	for i := 1; i < len(nums); i++ {
		for j := 0; j <= bag; j++ {
			dp[i][j] = dp[i-1][j]
			if j >= nums[i] {
				dp[i][j] += dp[i-1][j-nums[i]]
			}
		}
	}
	return dp[len(nums)-1][bag]
}
