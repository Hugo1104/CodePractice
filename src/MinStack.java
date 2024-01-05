import java.util.ArrayList;
import java.util.List;

public class MinStack {

    private class MinStackNode {
        int val, min;
        MinStackNode next;

        private MinStackNode(int val, int min, MinStackNode next) {
            this.val = val;
            this.min = min;
            this.next = next;
        }
    }

    private MinStackNode head;

    public MinStack() {
        // do initialization if necessary
    }

    /*
     * @param number: An integer
     * @return: nothing
     */
    public void push(int number) {
        // write your code here
        if (head == null) {
            head = new MinStackNode(number, number, null);
        } else {
            head = new MinStackNode(number, Math.min(number, head.min), head);
        }
    }

    /*
     * @return: An integer
     */
    public int pop() {
        // write your code here
        int num = head.val;
        head = head.next;
        return num;
    }

    /*
     * @return: An integer
     */
    public int min() {
        // write your code here
        return head.min;
    }
}