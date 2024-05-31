package main

func pathSum1(root *TreeNode, targetSum int) (ans [][]int) {
	var dfs func(node *TreeNode, sum int, path []int)
	dfs = func(node *TreeNode, sum int, path []int) {
		if node == nil {
			return
		}
		path = append(path, node.Val)
		sum += node.Val
		if node.Left == nil && node.Right == nil {
			if sum == targetSum {
				ans = append(ans, append([]int{}, path...))
			}
		}
		dfs(node.Left, sum, path)
		dfs(node.Right, sum, path)
	}
	dfs(root, 0, []int{})
	return
}

func hasPathSum1(root *TreeNode, targetSum int) bool {
	var dfs func(node *TreeNode, sum int) bool
	dfs = func(node *TreeNode, sum int) bool {
		if node == nil {
			return false
		}
		sum += node.Val
		if node.Left == nil && node.Right == nil && sum == targetSum {
			return true
		}
		if dfs(node.Left, sum) || dfs(node.Right, sum) {
			return true
		}
		return false
	}
	return dfs(root, 0)
}
func kConcatenationMaxSum1(arr []int, k int) int {
	var maxSubArray func(nums []int) int
	maxSubArray = func(nums []int) int {
		ans := nums[0]
		for i := 1; i < len(nums); i++ {
			nums[i] = max(nums[i], (nums[i-1]+nums[i])%(1e9+7))
			ans = max(ans, nums[i])
		}
		return max(ans, 0)
	}
	newArr := make([]int, 0, 3*len(arr))
	for ; k > 0; k-- {
		newArr = append(newArr, arr...)
	}
	return maxSubArray(newArr)
}
func removeElement1(nums []int, val int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		for i < n && nums[i] == val {
			nums[i], nums[n-1] = nums[n-1], nums[i]
			n--
		}
	}
	return n
}
func removeDuplicates1(nums []int) int {
	index, times := 1, 1
	for i := 1; i < len(nums); i++ {
		if nums[i] == nums[index-1] {
			times++
			if times <= 2 {
				nums[index] = nums[i]
				index++
			}
		} else {
			times = 1
			nums[index] = nums[i]
			index++
		}
	}
	return index
}
