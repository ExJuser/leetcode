package main

import (
	"fmt"
)

// SelectionSort 选择排序
// 每一轮都找到一个最小的数
func SelectionSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		//从i位置开始找到最小的数
		minIndex := i
		for j := i + 1; j < len(nums); j++ {
			if nums[j] < nums[minIndex] {
				minIndex = j
			}
		}
		//和当前位置做交换
		nums[i], nums[minIndex] = nums[minIndex], nums[i]
	}
}

// BubbleSort 冒泡排序
// 每一轮都将一个最大的数上浮到数组末尾 两两比较
func BubbleSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		for j := 0; j < len(nums)-i-1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
			}
		}
	}
}

// InsertSort 插入排序
// 三傻排序中最有用的 两两比较
func InsertSort(nums []int) {
	for i := 0; i < len(nums); i++ {
		//for j := i - 1; j >= 0; j-- {
		//	if nums[j+1] < nums[j] {
		//		nums[j], nums[j+1] = nums[j+1], nums[j]
		//	}
		//}
		for j := i; j >= 1; j-- {
			if nums[j] < nums[j-1] {
				nums[j-1], nums[j] = nums[j], nums[j-1]
			}
		}
	}
}

// Search 基础版二分搜索
// slices.BinarySearch
func Search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		middle := (right-left)/2 + left
		if nums[middle] == target {
			return middle
		} else if nums[middle] > target {
			right = middle - 1
		} else {
			left = middle + 1
		}
	}
	return -1
}

// FindLeft 找到第一个大于等于target的数 二分答案法中常用
// sort.Search
func FindLeft(nums []int, target int) int {
	left, right := 0, len(nums)-1
	ans := -1
	for left <= right {
		middle := (right-left)/2 + left
		if nums[middle] >= target {
			ans = middle
			right = middle - 1
		} else {
			left = middle + 1
		}
	}
	return ans
}

func MergeSort(nums []int) {
	mergeSort(nums, 0, len(nums)-1)
}

func mergeSort(nums []int, left, right int) {
	if left >= right {
		return
	}
	mid := (right-left)/2 + left
	mergeSort(nums, left, mid)
	mergeSort(nums, mid+1, right)
	merge(nums, left, mid, right)
}

func merge(nums []int, left, mid, right int) {
	temp := make([]int, right-left+1)
	i, j, k := left, mid+1, 0
	for i <= mid && j <= right {
		if nums[i] <= nums[j] {
			temp[k] = nums[i]
			i++
		} else {
			temp[k] = nums[j]
			j++
		}
		k++
	}
	for i <= mid {
		temp[k] = nums[i]
		k++
		i++
	}
	for j <= right {
		temp[k] = nums[j]
		k++
		j++
	}
	for i := 0; i < len(temp); i++ {
		nums[left+i] = temp[i]
	}
}

func main() {
	nums := []int{1, 5, 6, 7, 2, 6, 4, 5, 9}
	MergeSort(nums)
	fmt.Println(nums)
}
