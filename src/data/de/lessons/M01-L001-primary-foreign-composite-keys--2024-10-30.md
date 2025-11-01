# Primary Keys, Foreign Keys, and Composite Keys in Database Systems

## **What Are Database Keys?**

Database keys are fundamental constraints that ensure data integrity, establish relationships between tables, and enable efficient data retrieval. They serve as the backbone of relational database design, particularly critical in data warehousing and analytics environments.

**Key Types Covered:**

- **Primary Keys** - Unique identifiers for table rows
- **Foreign Keys** - Relationship connectors between tables  
- **Composite Keys** - Multi-column unique identifiers

These key concepts are essential for data engineers working with OLTP systems, data warehouses, and dimensional modeling in platforms like Snowflake, where proper key design directly impacts query performance and data quality.

------

## **1. Primary Keys: "The Unique Row Identifier"**

### Theory

A **Primary Key** is a column (or combination of columns) that uniquely identifies each row in a table. Every table should have exactly one primary key, and it must satisfy two critical requirements:

1. **Uniqueness** - No two rows can have the same primary key value
2. **Non-null** - Primary key values cannot be NULL or empty

The primary key serves as the "address" for each row, enabling fast lookups and ensuring referential integrity across the database.

### Why It Matters in Data Engineering

In data warehousing and ETL processes, primary keys are crucial for:
- **Upsert operations** (UPDATE or INSERT logic)
- **Change Data Capture (CDC)** tracking
- **Slowly Changing Dimension (SCD)** implementations
- **Data quality validation** and duplicate detection
- **Query optimization** through automatic indexing

### Real-World Examples

**Simple Primary Key - Customer Table:**

```sql
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,           -- Natural key
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample data
INSERT INTO customers VALUES 
    (1001, 'john.doe@email.com', 'John', 'Doe', '2024-01-15'),
    (1002, 'jane.smith@email.com', 'Jane', 'Smith', '2024-01-16');
```

**Surrogate Key Pattern (Common in Data Warehouses):**

```sql
CREATE TABLE dim_products (
    product_key INT IDENTITY(1,1) PRIMARY KEY,  -- Surrogate key
    product_id VARCHAR(50),                     -- Natural/business key
    product_name VARCHAR(200),
    category VARCHAR(100),
    valid_from DATE,
    valid_to DATE,
    is_current BOOLEAN DEFAULT TRUE
);
```

**What Happens:**
- `customer_id` automatically creates a unique clustered index
- Database enforces uniqueness - duplicate inserts will fail
- Fast lookups: `WHERE customer_id = 1001` uses index scan, not table scan
- ETL processes can reliably identify and update specific customer records

------

## **2. Foreign Keys: "The Relationship Builder"**

### Theory

A **Foreign Key** is a column (or combination of columns) that creates a link between two tables by referencing the primary key of another table. Foreign keys enforce **referential integrity**, ensuring that relationships between tables remain consistent.

**Foreign Key Rules:**
- Must reference a primary key or unique key in another table
- Values must exist in the referenced table (or be NULL if allowed)
- Prevents "orphaned" records that reference non-existent parent records

### Why It Matters in Data Engineering

Foreign keys are essential for:
- **Dimensional modeling** (fact tables referencing dimension tables)
- **Data lineage** and relationship mapping
- **ETL validation** - ensuring lookup values exist before loading
- **Query optimization** - join performance and execution plans
- **Data quality** - preventing referential integrity violations

### Real-World Examples

**E-commerce Order System:**

```sql
-- Parent table (referenced)
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100)
);

-- Child table (referencing)
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,                    -- Foreign key
    order_date DATE,
    total_amount DECIMAL(10,2),
    
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Sample data
INSERT INTO customers VALUES 
    (1001, 'john@email.com', 'John', 'Doe'),
    (1002, 'jane@email.com', 'Jane', 'Smith');

INSERT INTO orders VALUES 
    (50001, 1001, '2024-01-20', 299.99),    -- Valid: customer 1001 exists
    (50002, 1002, '2024-01-21', 149.50);    -- Valid: customer 1002 exists

-- This would FAIL due to foreign key constraint
INSERT INTO orders VALUES 
    (50003, 9999, '2024-01-22', 99.99);     -- Error: customer 9999 doesn't exist
```

**Data Warehouse Fact Table Example:**

```sql
-- Dimension tables
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,      -- Surrogate key
    customer_id VARCHAR(50),          -- Business key
    customer_name VARCHAR(200)
);

CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,       -- Surrogate key
    product_id VARCHAR(50),           -- Business key
    product_name VARCHAR(200)
);

-- Fact table with multiple foreign keys
CREATE TABLE fact_sales (
    sale_id INT PRIMARY KEY,
    customer_key INT,                 -- FK to dim_customer
    product_key INT,                  -- FK to dim_product
    sale_date DATE,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key)
);
```

**What Happens:**
- Database prevents inserting orders for non-existent customers
- Joins between tables are optimized using foreign key relationships
- ETL processes can validate data integrity before loading
- Query planners use foreign key information for better execution plans

------

## **3. Composite Keys: "Multi-Column Uniqueness"**

### Theory

A **Composite Key** (also called a compound key) is a primary key or unique key that consists of **multiple columns**. The combination of all columns in the composite key must be unique, even though individual columns may contain duplicate values.

**Composite Key Characteristics:**
- Requires **all columns together** to be unique
- Individual columns can have duplicate values
- Commonly used when no single column can uniquely identify a row
- Often used in many-to-many relationship bridge tables

### Why It Matters in Data Engineering

Composite keys are crucial for:
- **Bridge tables** in dimensional modeling
- **Time-series data** where uniqueness depends on entity + timestamp
- **Many-to-many relationships** (student-course, product-category)
- **Event tracking** where uniqueness combines multiple dimensions
- **Partitioning strategies** in large data warehouses

### Real-World Examples

**Student Course Enrollment (Many-to-Many Bridge):**

```sql
-- Individual tables
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(200),
    email VARCHAR(255)
);

CREATE TABLE courses (
    course_id VARCHAR(10) PRIMARY KEY,
    course_name VARCHAR(200),
    credits INT
);

-- Bridge table with composite primary key
CREATE TABLE student_enrollments (
    student_id INT,
    course_id VARCHAR(10),
    semester VARCHAR(20),
    enrollment_date DATE,
    grade VARCHAR(2),
    
    -- Composite primary key (all three columns together must be unique)
    PRIMARY KEY (student_id, course_id, semester),
    
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- Sample data
INSERT INTO student_enrollments VALUES 
    (1001, 'CS101', '2024-Spring', '2024-01-15', 'A'),
    (1001, 'CS102', '2024-Spring', '2024-01-15', 'B+'),   -- Same student, different course
    (1002, 'CS101', '2024-Spring', '2024-01-16', 'A-'),   -- Different student, same course
    (1001, 'CS101', '2024-Fall', '2024-08-20', 'A+');     -- Same student+course, different semester

-- This would FAIL - duplicate composite key
INSERT INTO student_enrollments VALUES 
    (1001, 'CS101', '2024-Spring', '2024-01-20', 'B');    -- Error: combination already exists
```

**Time-Series IoT Sensor Data:**

```sql
CREATE TABLE sensor_readings (
    sensor_id VARCHAR(50),
    timestamp TIMESTAMP,
    temperature DECIMAL(5,2),
    humidity DECIMAL(5,2),
    battery_level INT,
    
    -- Composite primary key: each sensor can only have one reading per timestamp
    PRIMARY KEY (sensor_id, timestamp)
);

-- Sample data
INSERT INTO sensor_readings VALUES 
    ('TEMP_001', '2024-01-20 10:00:00', 23.5, 65.2, 85),
    ('TEMP_001', '2024-01-20 10:01:00', 23.6, 65.1, 85),  -- Same sensor, different time
    ('TEMP_002', '2024-01-20 10:00:00', 22.1, 68.5, 92),  -- Different sensor, same time
    ('TEMP_002', '2024-01-20 10:01:00', 22.3, 68.2, 92);
```

**Data Warehouse Slowly Changing Dimension Type 2:**

```sql
CREATE TABLE dim_customer_scd2 (
    customer_id VARCHAR(50),          -- Business key
    effective_date DATE,              -- When this version became active
    customer_name VARCHAR(200),
    email VARCHAR(255),
    phone VARCHAR(20),
    end_date DATE,                    -- When this version became inactive
    is_current BOOLEAN,
    
    -- Composite primary key: customer + effective date must be unique
    PRIMARY KEY (customer_id, effective_date)
);

-- Track customer changes over time
INSERT INTO dim_customer_scd2 VALUES 
    ('CUST_001', '2024-01-01', 'John Doe', 'john@old-email.com', '555-1234', '2024-06-30', FALSE),
    ('CUST_001', '2024-07-01', 'John Doe', 'john@new-email.com', '555-1234', NULL, TRUE);
```

**What Happens:**
- Each combination of composite key columns must be unique
- Individual columns can have duplicates (sensor_id repeats, timestamp repeats)
- Database automatically creates a compound index on all key columns
- ETL processes can safely upsert based on the complete composite key

------

## **How Database Keys Work Together**

### Complete Example: E-commerce Data Model

```sql
-- Customer dimension (primary key)
CREATE TABLE dim_customer (
    customer_key INT IDENTITY(1,1) PRIMARY KEY,    -- Surrogate key
    customer_id VARCHAR(50) UNIQUE,                -- Natural key
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255)
);

-- Product dimension (primary key)
CREATE TABLE dim_product (
    product_key INT IDENTITY(1,1) PRIMARY KEY,     -- Surrogate key
    product_id VARCHAR(50) UNIQUE,                 -- Natural key
    product_name VARCHAR(200),
    category VARCHAR(100),
    unit_price DECIMAL(10,2)
);

-- Orders fact table (primary key + foreign keys)
CREATE TABLE fact_orders (
    order_id INT PRIMARY KEY,                      -- Simple primary key
    customer_key INT,                              -- Foreign key to dim_customer
    order_date DATE,
    total_amount DECIMAL(12,2),
    
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key)
);

-- Order details bridge table (composite key + foreign keys)
CREATE TABLE bridge_order_details (
    order_id INT,                                  -- Foreign key to fact_orders
    product_key INT,                               -- Foreign key to dim_product
    line_number INT,                               -- Part of composite key
    quantity INT,
    unit_price DECIMAL(10,2),
    line_total DECIMAL(12,2),
    
    -- Composite primary key: order + product + line must be unique
    PRIMARY KEY (order_id, product_key, line_number),
    
    FOREIGN KEY (order_id) REFERENCES fact_orders(order_id),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key)
);
```

------

## **Practice Exercises**

### Exercise 1: Identify Key Types

For each scenario, identify what type of key constraint is needed:

1. **Scenario A:** A table storing unique social security numbers for employees.
   - **Key Type:** _________
2. **Scenario B:** A table linking students to the courses they're enrolled in this semester.
   - **Key Type:** _________  
3. **Scenario C:** An order table that must reference valid customers.
   - **Key Type:** _________
4. **Scenario D:** A sensor data table where each device can only have one reading per minute.
   - **Key Type:** _________

### Exercise 2: Design Key Constraints

Design the appropriate key constraints for these scenarios:

1. **Library System:** Books, Authors, and the many-to-many relationship between them
2. **Sales Analytics:** Customer orders with multiple products per order
3. **Time Series Data:** Stock prices with symbol and timestamp combinations

### Exercise 3: Identify Key Violations

Which of these would violate key constraints and why?

```sql
-- Given this table structure:
CREATE TABLE product_reviews (
    product_id VARCHAR(50),
    user_id VARCHAR(50), 
    review_date DATE,
    rating INT,
    review_text TEXT,
    PRIMARY KEY (product_id, user_id, review_date)
);

-- Which inserts would fail?
INSERT INTO product_reviews VALUES ('PROD_001', 'USER_123', '2024-01-20', 5, 'Great product!');
INSERT INTO product_reviews VALUES ('PROD_001', 'USER_456', '2024-01-20', 4, 'Good quality');
INSERT INTO product_reviews VALUES ('PROD_001', 'USER_123', '2024-01-20', 3, 'Changed my mind');
INSERT INTO product_reviews VALUES ('PROD_002', 'USER_123', '2024-01-20', 5, 'Love it!');
```

------

## **Key Takeaways**

✅ **Primary Keys** = Unique row identifiers (essential for ETL and CDC)
✅ **Foreign Keys** = Relationship enforcers (critical for dimensional modeling)  
✅ **Composite Keys** = Multi-column uniqueness (perfect for bridge tables and time-series)

### Design Best Practices

- **Use surrogate keys** in data warehouses for better performance and SCD handling
- **Implement foreign keys** to catch data quality issues early in ETL processes
- **Choose composite keys** when business logic requires multi-column uniqueness
- **Index foreign key columns** for better join performance
- **Document key relationships** for data lineage and impact analysis

### When to Apply Each Type

- **Primary Keys:** Every table needs one (use surrogate keys in dimensional models)
- **Foreign Keys:** Any table referencing another table (fact → dimension relationships)
- **Composite Keys:** Bridge tables, time-series data, event tracking, many-to-many relationships

------

## **Next Steps**

1. Practice creating tables with different key types using the examples
2. Try designing a complete dimensional model with proper key relationships
3. Research how Snowflake handles key constraints and clustering keys
4. Explore how dbt models can implement and test key relationships
5. Learn about surrogate key generation strategies for data warehouses
