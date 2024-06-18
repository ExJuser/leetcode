package main

import (
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
//				helper(left, j)
//				helper(i, right)
//			}
//		}
//		helper(0, len(nums)-1)
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

// 冒泡排序
func bubbleSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}
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
