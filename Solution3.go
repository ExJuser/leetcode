package main

import (
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// 贪心算法分发饼干
// 每个孩子只给一块
func findContentChildren(g []int, s []int) int {
	slices.Sort(g)
	slices.Sort(s)
	var i, j int
	for ; i < len(g); i++ {
		index := sort.SearchInts(s[j:], g[i])
		if index == len(s)-j {
			return i
		} else {
			j += index + 1
		}
	}
	return i
}

// 二叉树的所有路径
func binaryTreePaths(root *TreeNode) (ans []string) {
	var dfs func(node *TreeNode, path []string)
	dfs = func(node *TreeNode, path []string) {
		if node == nil {
			return
		}
		path = append(path, strconv.Itoa(node.Val))
		if node.Left == nil && node.Right == nil {
			ans = append(ans, strings.Join(path, "->"))
		}
		dfs(node.Left, path)
		dfs(node.Right, path)
	}
	dfs(root, []string{})
	return
}

// 找出所有从根节点到叶子节点路径总和等于给定目标和的路径。
func pathSum(root *TreeNode, targetSum int) (ans [][]int) {
	var dfs func(node *TreeNode, sum int, path []int)
	dfs = func(node *TreeNode, sum int, path []int) {
		if node == nil {
			return
		}
		path = append(path, node.Val)
		sum += node.Val
		if node.Left == nil && node.Right == nil && sum == targetSum {
			ans = append(ans, append([]int{}, path...))
		}
		dfs(node.Left, sum, path)
		dfs(node.Right, sum, path)
	}
	dfs(root, 0, []int{})
	return
}

// 字母大小写全排列
func letterCasePermutation(s string) (ans []string) {
	var dfs func(index int, path []byte)
	dfs = func(index int, path []byte) {
		if index == len(s) {
			ans = append(ans, string(path))
			return
		}
		ch := rune(s[index])
		if unicode.IsLetter(ch) {
			dfs(index+1, append(path, byte(unicode.ToLower(ch))))
			dfs(index+1, append(path, byte(unicode.ToUpper(ch))))
		}
	}
	dfs(0, []byte{})
	return
}

// 烹饪料理
func perfectMenu(materials []int, cookbooks [][]int, attribute [][]int, limit int) int {
	/**
	materials[j] 表示第 j 种食材的数量
	cookbooks[i][j] 表示制作第 i 种料理需要第 j 种食材的数量
	attribute[i] = [x,y] 表示第 i 道料理的美味度 x 和饱腹感 y
	饱腹感不小于 limit 的情况下，请返回勇者可获得的最大美味度
	*/
	ans := -1
	var dfs func(menuIndex, full, yummy int)
	dfs = func(menuIndex, full, yummy int) {
		//做完了所有的料理
		if menuIndex == len(cookbooks) {
			//饱腹感达到要求
			if full >= limit {
				//更新最大美味度
				ans = max(ans, yummy)
			}
			return
		}
		flag := true
		for j := 0; j < len(cookbooks[menuIndex]); j++ {
			if materials[j] < cookbooks[menuIndex][j] {
				flag = false
				break
			}
		}
		if flag {
			for j := 0; j < len(cookbooks[menuIndex]); j++ {
				materials[j] -= cookbooks[menuIndex][j]
			}
			dfs(menuIndex+1, full+attribute[menuIndex][1], yummy+attribute[menuIndex][0])
			for j := 0; j < len(cookbooks[menuIndex]); j++ {
				materials[j] += cookbooks[menuIndex][j]
			}
		}
		dfs(menuIndex+1, full, yummy)
	}
	dfs(0, 0, 0)
	return ans
}

// 摆动序列 动态规划解法
func wiggleMaxLength(nums []int) int {
	dp1 := make([]int, len(nums))
	dp2 := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		dp1[i] = 1
		dp2[i] = 1
	}
	for i := 1; i < len(nums); i++ {
		for j := 0; j < i; j++ {
			if dp1[j]%2 == 0 && nums[i] < nums[j] || dp1[j]%2 != 0 && nums[i] > nums[j] {
				dp1[i] = max(dp1[i], dp1[j]+1)
			}
			if dp2[j]%2 == 0 && nums[i] > nums[j] || dp2[j]%2 != 0 && nums[i] < nums[j] {
				dp2[i] = max(dp2[i], dp2[j]+1)
			}
		}
	}
	return max(slices.Max(dp1), slices.Max(dp2))
}
