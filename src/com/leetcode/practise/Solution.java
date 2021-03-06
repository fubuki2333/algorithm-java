package com.leetcode.practise;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:Solution
 * Package:com.leetcode.practise
 * Description:
 *
 * @Date:2020/7/6 下午 11:17
 * @Author:gaochenyu2020@163.com
 */
public class Solution {

    /**
     * Z字形变换
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {

        if (numRows == 1)
            return s;

        List<StringBuilder> rows = new ArrayList<>();
        for(int i = 0;i < Math.min(numRows,s.length());i++)
            rows.add(new StringBuilder());

        int curRow = 0;
        boolean goingDown = false;

        for (char c : s.toCharArray()){
            rows.get(curRow).append(c);
            if (curRow == 0 || curRow == numRows - 1)
                goingDown = !goingDown;
            curRow += goingDown ? 1 : -1;
        }

        StringBuilder ret = new StringBuilder();
        for (StringBuilder row : rows) {
            ret.append(row);
        }
        return ret.toString();
    }

    /**
     * 三数之和
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        int len = nums.length;
        List<List<Integer>> result = new ArrayList<>();
        if (len < 3)
            return result;
        Arrays.sort(nums);
        // 枚举first
        for (int first = 0; first < len; first++){
            if (first > 0 && nums[first] == nums[first-1]) {
                continue;
            }
            // third从尾部向前枚举
            int third = len - 1;
            // 枚举second
            for (int second = first + 1; second < len; second++){
                if (second > first + 1 && nums[second] == nums[second-1]){
                    continue;
                }
                while (second < third && nums[first] + nums[second] + nums[third] > 0){
                    third--;
                }
                if (second == third){
                    break;
                }
                if (nums[first] + nums[second] + nums[third] == 0){
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    result.add(list);
                }
            }
        }
        return result;
    }

    /**
     * 整数转罗马数字
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] symbols = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        StringBuilder sb = new StringBuilder();
        for (int i = 0;i < values.length && num >= 0;i++){
            while (values[i] <= num){
                num -= values[i];
                sb.append(symbols[i]);
            }
        }
        return sb.toString();
    }


    /**
     * 图像渲染
     * @param image
     * @param sr
     * @param sc
     * @param newColor
     * @return
     */
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int currentColor = image[sr][sc];
        if (currentColor == newColor)
            return image;

        int[] dx = {1, 0, 0, -1};
        int[] dy = {0, 1, -1, 0};

        int n = image.length,m = image[0].length;
        Queue<int[]> queue = new LinkedList<int[]>();
        queue.offer(new int[]{sr,sc});
        image[sr][sc] = newColor;
        while (!queue.isEmpty()){
            int[] cell = queue.poll();
            int x = cell[0],y = cell[1];
            for (int i = 0; i < 4; i++) {
                int mx = x + dx[i], my = y + dy[i];
                if (mx >= 0 && mx < n && my >= 0 && my < m && image[mx][my] == currentColor) {
                    queue.offer(new int[]{mx, my});
                    image[mx][my] = newColor;
                }
            }
        }
        return image;
    }

}
