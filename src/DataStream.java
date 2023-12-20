import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

public class DataStream {
    HashMap<Integer, Integer> map;
    Queue<Integer> queue;

    public DataStream(){
        // do initialization if necessary
        map = new HashMap<>();
        queue = new LinkedList<>();
    }
    /**
     * @param num: next number in stream
     * @return: nothing
     */
    public void add(int num) {
        // write your code here
        queue.add(num);
        map.put(num, map.getOrDefault(num, 0) + 1);

    }

    /**
     * @return: the first unique number in stream
     */
    public int firstUnique() {
        // write your code here
        while (!queue.isEmpty() && map.get(queue.peek()) > 1) {
            queue.poll();
        }

        if (queue.isEmpty()) {
            return -1;
        } else {
            return queue.peek();
        }
    }
}