import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;
import java.util.stream.Collectors;

class LionDescription {
    public String name;
    public int height;
}

class LionSchedule {
    public String name;
    public int enterTime;
    public int exitTime;
}

class LionCompetition {
    private Map<String, Integer> myLionsInRoom;
    private TreeMap<Integer, Integer> heightsInRoom;
    private Map<String, LionDescription> lionDescriptions;
    private Map<String, LionSchedule> lionSchedules;

    public LionCompetition(List<LionDescription> lions, List<LionSchedule> schedule) {
        myLionsInRoom = new HashMap<>();
        heightsInRoom = new TreeMap<>();
        lionDescriptions = new HashMap<>();
        lionSchedules = new HashMap<>();

        for (LionDescription ld : lions) {
            lionDescriptions.put(ld.name, ld);
        }

        for (LionSchedule ls : schedule) {
            lionSchedules.put(ls.name, ls);
        }
    }

    public void lionEntered(int currentTime, int height) {
        // Check if any of our lions are scheduled to enter at this time.
        for (Map.Entry<String, LionSchedule> entry : lionSchedules.entrySet()) {
            if (entry.getValue().enterTime == currentTime) {
                myLionsInRoom.put(entry.getKey(), lionDescriptions.get(entry.getKey()).height);
            }
        }

        heightsInRoom.put(height, heightsInRoom.getOrDefault(height, 0) + 1);
    }

    public void lionLeft(int currentTime, int height) {
        // Check if any of our lions are scheduled to exit at this time.
        for (Map.Entry<String, LionSchedule> entry : lionSchedules.entrySet()) {
            if (entry.getValue().exitTime == currentTime) {
                myLionsInRoom.remove(entry.getKey());
            }
        }

        int count = heightsInRoom.get(height);
        if (count == 1) {
            heightsInRoom.remove(height);
        } else {
            heightsInRoom.put(height, count - 1);
        }
    }

    public List<String> getBiggestLions() {
        List<String> result = new ArrayList<>();
        if (heightsInRoom.isEmpty()) return result;

        int maxOtherHeight = heightsInRoom.lastKey();

        for (Map.Entry<String, Integer> entry : myLionsInRoom.entrySet()) {
            if (entry.getValue() >= maxOtherHeight) {
                result.add(entry.getKey());
            }
        }

        Collections.sort(result); // Sort the lion names alphabetically.
        return result;
    }
}

public class Solution {
    public static void main(String args[]) throws Exception {
        Scanner scanner = new Scanner(System.in);
        String operation;

        List<LionDescription> descriptions = new ArrayList<LionDescription>();
        List<LionSchedule> schedule = new ArrayList<LionSchedule>();

        do
        {
            operation = scanner.next();

            if (operation.equals("definition"))
            {
                LionDescription description = new LionDescription();
                description.name = scanner.next();
                description.height = scanner.nextInt();

                descriptions.add(description);
            }
            if (operation.equals("schedule"))
            {
                LionSchedule scheduleEntry = new LionSchedule();
                scheduleEntry.name = scanner.next();
                scheduleEntry.enterTime = scanner.nextInt();
                scheduleEntry.exitTime = scanner.nextInt();

                schedule.add(scheduleEntry);
            }
        } while (!operation.equals("start"));

        LionCompetition lionCompetition = new LionCompetition(descriptions, schedule);

        do
        {
            int currentTime = scanner.nextInt();
            operation = scanner.next();

            if (operation.equals("enter"))
            {
                int size = scanner.nextInt();

                lionCompetition.lionEntered(currentTime, size);
            }
            if (operation.equals("exit"))
            {
                int size = scanner.nextInt();

                lionCompetition.lionLeft(currentTime, size);
            }
            if (operation.equals("inspect"))
            {
                List<String> lions = lionCompetition.getBiggestLions();

                System.out.print(lions.size());

                for (String name : lions) {
                    System.out.print(" " + name);
                }

                System.out.println();
            }
        } while (!operation.equals("end"));
    }
}