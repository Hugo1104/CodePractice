import java.util.LinkedList;
import java.util.List;

public class ConcertTickets {

    static class Requirement {
        int artistId;
        int locationId;
        int ticketPrice;
        int category;
        int availableSeats;

        Requirement(int artistId, int locationId, int ticketPrice, int category, int availableSeats) {
            this.artistId = artistId;
            this.locationId = locationId;
            this.ticketPrice = ticketPrice;
            this.category = category;
            this.availableSeats = availableSeats;
        }
    }

    private List<Requirement> requirements = new LinkedList<>();

    public void OnNewRequirement(int artist, int location, int ticketPrice, int category, int availableSeats) {
        requirements.add(new Requirement(artist, location, ticketPrice, category, availableSeats));
    }

    public int ProcessData(int messageId, int[] data) {
        // Check if it's a valid message
        if (messageId == 0) {
            return 0;
        }

        // Extract details from the packet
        int artistId = data[0];
        int locationId = data[1];
        int ticketPrice = data[2];
        int category = data[3];
        int availableSeats = data[4];

        // Check requirements for a match
        for (Requirement req : requirements) {
            if (req.artistId == artistId
                    && req.locationId == locationId
                    && req.ticketPrice >= ticketPrice
                    && req.category <= category
                    && req.availableSeats <= availableSeats) {
                requirements.remove(req);
                return messageId; // order message reference
            }
        }

        return 0; // no order to send
    }

    public static void main(String[] args) {
        ConcertTickets concertTickets = new ConcertTickets();

        // Test based on the sample provided
        concertTickets.OnNewRequirement(1, 1, 100, 1, 2);
        int[][] data = {
                {1, 1, 110, 1, 2},
                {1, 1, 90, 1, 2},
                {1, 1, 100, 1, 3}
        };
        for (int i = 0; i < data.length; i++) {
            System.out.println(concertTickets.ProcessData(i + 1, data[i]));
        }
    }
}
