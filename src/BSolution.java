import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class BSolution {
}
class sSolution {
    /**
     * This method will be invoked first, you should design your own algorithm
     * to serialize a binary tree which denote by a root node to a string which
     * can be easily deserialized by your own "deserialize" method later.
     */
    public String serialize(TreeNode root) {
        // write your code here
        return serializeHelper(root, "");
    }

    /**
     * This method will be invoked second, the argument data is what exactly
     * you serialized at method "serialize", that means the data is not given by
     * system, it's given by your own serialize method. So the format of data is
     * designed by yourself, and deserialize it here as you serialize it in
     * "serialize" method.
     */
    public TreeNode deserialize(String data) {
        // write your code here
        String[] dataList = data.split(",");
        List<String> list = new LinkedList<>(Arrays.asList(dataList));
        return deserializeHelper(list);
    }

    private String serializeHelper(TreeNode node, String str) {
        if (node == null) {
            str += "None,";
        } else {
            str += String.valueOf(node.val) + ",";
            str = serializeHelper(node.left, str);
            str = serializeHelper(node.right, str);
        }

        return str;
    }

    private TreeNode deserializeHelper(List<String> list) {
        if (list.get(0).equals("None")) {
            list.remove(0);
            return null;
        }

        TreeNode node = new TreeNode(Integer.parseInt(list.get(0)));
        list.remove(0);
        node.left = deserializeHelper(list);
        node.right =deserializeHelper(list);

        return node;
    }
}
