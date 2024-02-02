<!DOCTYPE html>
<html>
<head>
	<title>Delicious Restaurant</title>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<style>
		body {
			font-family: Arial, sans-serif;
			margin: 0;
			padding: 0;
		}
		header {
			background-color: #3e2609;
			padding: 10px;
			text-align: center;
		}
		h1 {
			margin: 0;
			font-size: 36px;
			color: #fff;
			text-shadow: 2px 2px #000;
		}
		nav {
			background-color: #333;
			color: #fff;
			display: flex;
			justify-content: space-around;
			align-items: center;
			padding: 10px;
		}
		nav a {
			color: #fff;
			text-decoration: none;
			font-size: 20px;
		}
		nav a:hover {
			color: #09ceff;
		}
		section {
			display: flex;
			flex-wrap: wrap;
			padding: 20px;
		}
		article {
			flex-basis: 400px;
			margin: 20px;
			background-color: #fff;
			box-shadow: 2px 2px 5px #ccc;
			border-radius: 5px;
			overflow: hidden;
		}
		article img {
			width: 100%;
			height: 200px;
			object-fit: cover;
		}
		article h2 {
			margin: 10px;
			font-size: 24px;
			color: #333;
		}
		article p {
			margin: 10px;
			font-size: 18px;
			color: #777;
			line-height: 1.5;
		}
		article button {
			display: block;
			margin: 10px auto;
			padding: 10px 20px;
			background-color: #f2b632;
			color: #fff;
			border: none;
			border-radius: 20px;
			font-size: 18px;
			cursor: pointer;
		}
		article button:hover {
			background-color: #333;
		}
		footer {
			background-color: #333;
			color: #fff;
			text-align: center;
			padding: 10px;
		}
		footer p {
			margin: 0;
			font-size: 16px;
		}
        table{
            border-collapse: collapse;
            width: 100%;
            text-align: center;
        }
        p{
            text-align: center;
        }
        
	</style>
</head>
<body>
	<header>
		<h1>Delicious Restaurant</h1>
	</header>
	<nav>
		<a href="restaurant.html">Home</a>
		<a href="empSchedule.php">Schedule</a>
		<a href="Orders.php">Orders</a>
		<a href="addCustomer.php">Add Customer</a>
        <a href="#">Order Summary</a>

	</nav>
    
    <?php
        
        //Connect to the database
        require_once 'connectDB.php';

        //Create a table that shows dates on which orders were placed along with the number of orders on that date.
        $sql = "SELECT timePlaced AS orderDate, COUNT(*) AS numOrders
                FROM orderPlaced
                GROUP BY timePlaced
                ORDER BY timePlaced";

        //Execute the query
        $result = $conn->query($sql);

        //display the results
        if ($result->rowCount() > 0) {
            echo "<table><tr><th>Date</th><th>Number of Orders</th></tr>";
            // output data of each row
            while($row = $result->fetch()) {
                echo "<tr><td>" . $row["orderDate"]. "</td><td>" . $row["numOrders"]. "</td></tr>";
            }
            echo "</table>";
        } else {
            echo "0 results";
        }

        

    ?>
</body>
</html>


   