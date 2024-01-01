import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;

public class MovingAverage {
    /*
     * @param size: An integer
     */

    long size, windowSum = 0, count = 0;
    Deque<Integer> queue = new ArrayDeque<>();
    public MovingAverage(int size) {
        // do intialization if necessary
        this.size = size;
    }

    /*
     * @param val: An integer
     * @return:
     */
    public double next(int val) {
        // write your code here
        count++;
        queue.add(val);
        int tail = count > size ? queue.poll() : 0;

        windowSum = windowSum + val - tail;

        return (double) windowSum / Math.min(size, count);
    }
}