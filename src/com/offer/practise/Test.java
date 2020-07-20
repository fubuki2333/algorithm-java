package com.offer.practise;

import java.util.ArrayList;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:Main
 * Package:com.offer.practise
 * Description:
 *
 * @Date:2020/2/29 下午 5:34
 * @Author:gaochenyu2020@163.com
 */
public class Test {

    public volatile static int i = 1;
    public static Lock lock = new ReentrantLock();

    public static void main(String[] args) {
//        Solution solution = new Solution();
//        int[] ints = new int[]{2,4,3,6,3,2,5,5};
//        int[] num1 = {};
//        int[] num2 = {};
//
//        solution.FindNumsAppearOnce(ints,num1,num2);
//        System.out.println(num1.toString());
//        System.out.println(num2.toString());
        test01();
    }

    public static void test01(){
        new Thread(()->{
            while (i < 100){
                synchronized (lock){
                    if (i % 2 == 1){
                        System.out.println(Thread.currentThread().getName()+"-----"+(i++));
                        lock.notifyAll();
                    }else {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }).start();
        new Thread(()->{
            while (i < 100){
                synchronized (lock){
                    if (i % 2 == 0){
                        System.out.println(Thread.currentThread().getName()+"-----"+(i++));
                        lock.notifyAll();
                    }else {
                        try {
                            lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }).start();
    }
}
