//import java.util.LinkedList;
//import java.util.Queue;
//
//public class Stack {
//    public Queue<Integer> queue1;
//    public Queue<Integer> queue2;
//
//    public Stack() {
//        queue1 = new LinkedList<>();
//        queue2 = new LinkedList<>();
//    }
//    /*
//     * @param x: An integer
//     * @return: nothing
//     */
//    public void push(int x) {
//        // write your code here
//        queue2.offer(x);
//        while (!queue1.isEmpty()) {
//            queue2.offer(queue1.poll());
//        }
//
//        Queue<Integer> temp = queue1;
//        queue1 = queue2;
//        queue2 = temp;
//    }
//
//    /*
//     * @return: nothing
//     */
//    public void pop() {
//        // write your code here
//        queue1.poll();
//    }
//
//    /*
//     * @return: An integer
//     */
//    public int top() {
//        // write your code here
//        return queue1.peek();
//    }
//
//    /*
//     * @return: True if the stack is empty
//     */
//    public boolean isEmpty() {
//        // write your code here
//        return queue1.isEmpty();
//    }
//}