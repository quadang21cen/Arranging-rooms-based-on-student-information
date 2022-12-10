DROP DATABASE IF EXISTS 'dormitory';


DROP TABLE IF EXISTS `dormitory`.`Student`;


CREATE TABLE `dormitory`.`Student` (
  `id` INT NOT NULL,
  `name` NVARCHAR(100) NULL,
  `gender` TINYINT NULL,
  `hometown` VARCHAR(100) NULL,
  `Bio_personality` NVARCHAR(500) NULL,
  `food_drink` NVARCHAR(500) NULL,
  `hob_inter` NVARCHAR(500) NULL,
  `smoking` NVARCHAR(100) NULL,
  `refer_roommate` NVARCHAR(500) NULL,
  `Cleanliess` TINYINT NULL,
  `Privacy` TINYINT NULL,
  PRIMARY KEY (`id`));

DROP TABLE IF EXISTS `dormitory`.`student_room`;

CREATE table `dormitory`.`student_room`(
	  `id` int not null,
    `room` int not null,
    FOREIGN KEY (`id`) REFERENCES student(`id`)
)



  
