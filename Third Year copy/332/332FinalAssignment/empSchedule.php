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
        form{
            text-align: center;

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
        table{

            width:100%;
            border: 1px solid black;
            text-align: center;

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
		<a href="#">Schedule</a>
		<a href="Orders.php">Orders</a>
		<a href="addCustomer.php">Add Customer</a>
        <a href="orderSum.php">Order Sumary</a>

	</nav>

   
    <form action="<?php echo $_SERVER['PHP_SELF'];?>" method="POST">
            <label for="employee">Employee:</label>
            <select id="employee" name="id" required>
                <option value="">Select an employee</option>
                <?php
                    // Connect to the database
                    require_once "connectDB.php";
                    $sql = "SELECT * FROM employee";
                    $stmt = $conn->prepare($sql);
                    $stmt->execute();

                    $result = $stmt->fetchAll();
                    foreach($result as $row){
                        echo "<option value='".$row['id']."'>".$row['name']."</option>";
                    }
                ?>
            </select>
            <input type="submit" name="submit" value="Submit">
    </form>
    <?php

    if(isset($_POST['submit'])){

        $id = $_POST['id'];
        // Connect to the database
        require_once "connectDB.php";


        $sql2 = "SELECT employee.name, weekday, start_time AS startT, end_time AS endT 
        FROM employeeSchedule
        JOIN employee 
        ON  employeeSchedule.employeeID = employee.id 
        WHERE employeeID = ? AND NOT(weekday = 'Saturday' OR weekday = 'Sunday')";

        $stmt = $conn->prepare($sql2);
        $stmt->execute([$id]);

        $weeklyShedule = [];
        while($row = $stmt->fetch(PDO::FETCH_ASSOC)){
            $weeklyShedule[$row['weekday']] = $row;
        }

        if(empty($weeklyShedule)){
            echo "<p>There is no schedule for this employee.</p>";
        }else{
            echo "<table>
                    <tr>
                        <th>Day</th>
                        <th>Start Time</th>
                        <th>End Time</th>
                    </tr>";

            $weekdays = array( "Monday", "Tuesday", "Wednesday", "Thursday", "Friday");

            for($i = 0; $i < count($weekdays); $i++){
                $day = $weekdays[$i];

                $start_str = "N/A";
                $end_str = "N/A";
                if (isset($weeklyShedule[$day])) {
                    $start_str = $weeklyShedule[$day]['startT'];
                    $end_str = $weeklyShedule[$day]['endT'];
                }

                echo "<tr>
                        <td>".$day."</td>
                        <td>".$start_str."</td>
                        <td>".$end_str."</td>
                    </tr>";
            }
            echo "</table>";
        }
    }
    ?>


</body>
</html>
   