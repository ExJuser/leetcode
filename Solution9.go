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

func quickSort_(nums []int) {
	var helper func(left, right int)
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
}
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
type MinStack struct {
}

//
//func Constructor() MinStack {
//
//}

func (this *MinStack) Push(val int) {

}

func (this *MinStack) Pop() {

}

func (this *MinStack) Top() int {

}

func (this *MinStack) GetMin() int {

}
