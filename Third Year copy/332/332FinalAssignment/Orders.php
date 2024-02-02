

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
		}
        h2{
            margin: 0;
			font-size: 36px;
			color: #333;
            text-align: center;
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
            text-align: center;
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
            text-align: center;
		}
		article button {
			display: block;
			margin: 10px auto;
            border: none;
			border-radius: 20px;
			font-size: 18px;
			cursor: pointer;
			padding: 10px 20px;
			background-color: #f2b632;
			color: #fff;
			
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
            text-align: center;
		}
        table {
        border-collapse: collapse;
        margin: 0 auto;
        }
        
        table:first-child {
        margin-bottom: 20px;
        }  
        form{
            text-align: center;
            
        }
        .date-input {
			display: flex;
			justify-content: center;
			align-items: center;
			padding: 10px;
		}
		.date-input label {
			font-size: 20px;
			color: #333;
			margin-right: 10px;
		}
		.date-input input[type="date"] {
			padding: 10px;
			font-size: 18px;
			color: #777;
			border-radius: 5px;
			border: none;
			outline: none;
			box-shadow: 2px 2px 5px #4c4b4b;
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
		<a href="#">Orders</a>
		<a href="addCustomer.php">Add Customer</a>
        <a href="orderSum.php">Order Summary</a>
	</nav>
	<h2>
        Orders by Date
    </h2>

    <form action="<?php echo $_SERVER['PHP_SELF'];?>" method="POST">
            <label for="date">Date:</label>
            <input type="date" name="date" id="date">
            <input type="submit" name="submit" value="Submit">
    </form>
    <?php
    

    if(isset($_POST['submit'])){

        require_once 'connectDB.php';

       $date = $_POST['date'];
       $sql = "SELECT customer.firstName, customer.lastName, orderPlaced.items, orderPlaced.timePlaced AS orderDate, orderPlaced.price as totalPrice, orderPlaced.tip, foodItem.name AS item, employee.name AS deliverer
       FROM orderPlaced
       JOIN customer ON orderPlaced.emailKey = customer.email
       JOIN foodItem ON orderPlaced.items = foodItem.name
       JOIN employee ON orderPlaced.deliverer = employee.id
       WHERE DATE(orderPlaced.timePlaced) = ?";    

        $stmt = $conn->prepare($sql);
        $stmt->execute([$date]);
        

        if($stmt->rowCount() > 0){
            echo "<table border='1'>
            <tr>
            <th>Customer Name</th>
            <th>Items</th>
            <th>Order Date</th>
            <th>Total Price</th>
            <th>Tip</th>
            <th>Delivery Person</th>
            </tr>";
            while($row = $stmt->fetch(PDO::FETCH_ASSOC)){
                echo "<tr>";
                echo "<td>" . $row['firstName'] . " " . $row['lastName'] . "</td>";
                echo "<td>" . $row['items'] . "</td>";
                echo "<td>" . $row['orderDate'] . "</td>";
                echo "<td>" . $row['totalPrice'] . "</td>";
                echo "<td>" . $row['tip'] . "</td>";
                echo "<td>" . $row['deliverer'] . "</td>";
                echo "</tr>";
            }
            echo "</table>";
        }else{
            echo "<p>No orders were made on that date.</p>";
        }
    }
    ?>
</body>


</html>