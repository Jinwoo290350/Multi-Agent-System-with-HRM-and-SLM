-- Decision Agent Database Schema
-- Run this file to create the complete database structure

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS task_history;
DROP TABLE IF EXISTS project_assignments;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS knowledge_documents;
DROP TABLE IF EXISTS system_logs;

-- Create departments table
CREATE TABLE departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    manager_id INTEGER,
    budget DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    full_name TEXT GENERATED ALWAYS AS (first_name || ' ' || last_name) STORED,
    department_id INTEGER,
    position TEXT,
    level TEXT, -- Junior, Mid, Senior, Lead, Manager, Director
    salary DECIMAL(10,2),
    hire_date DATE,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'terminated', 'on_leave')),
    phone TEXT,
    address TEXT,
    emergency_contact TEXT,
    skills TEXT, -- JSON array of skills
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (department_id) REFERENCES departments (id)
);

-- Create projects table
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'planning' CHECK (status IN ('planning', 'active', 'on_hold', 'completed', 'cancelled')),
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5), -- 1=Critical, 5=Low
    budget DECIMAL(15,2),
    actual_cost DECIMAL(15,2) DEFAULT 0,
    start_date DATE,
    end_date DATE,
    actual_end_date DATE,
    manager_id INTEGER,
    department_id INTEGER,
    client_name TEXT,
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage BETWEEN 0 AND 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (manager_id) REFERENCES users (id),
    FOREIGN KEY (department_id) REFERENCES departments (id)
);

-- Create tasks table
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_code TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled', 'blocked')),
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5), -- 1=Critical, 5=Low
    type TEXT DEFAULT 'development' CHECK (type IN ('development', 'testing', 'design', 'research', 'meeting', 'documentation', 'support')),
    project_id INTEGER,
    assigned_to INTEGER,
    created_by INTEGER,
    estimated_hours DECIMAL(5,2),
    actual_hours DECIMAL(5,2) DEFAULT 0,
    due_date DATE,
    completion_date DATE,
    tags TEXT, -- JSON array of tags
    dependencies TEXT, -- JSON array of task IDs this depends on
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects (id),
    FOREIGN KEY (assigned_to) REFERENCES users (id),
    FOREIGN KEY (created_by) REFERENCES users (id)
);

-- Create project assignments table (many-to-many)
CREATE TABLE project_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT DEFAULT 'member' CHECK (role IN ('manager', 'lead', 'member', 'observer')),
    allocation_percentage INTEGER DEFAULT 100 CHECK (allocation_percentage BETWEEN 0 AND 100),
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects (id),
    FOREIGN KEY (user_id) REFERENCES users (id),
    UNIQUE(project_id, user_id)
);

-- Create knowledge documents table
CREATE TABLE knowledge_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL, -- 'policy', 'procedure', 'guideline', 'faq', 'technical'
    subcategory TEXT,
    version TEXT DEFAULT '1.0',
    status TEXT DEFAULT 'published' CHECK (status IN ('draft', 'review', 'published', 'archived')),
    author_id INTEGER,
    reviewer_id INTEGER,
    tags TEXT, -- JSON array of tags
    access_level TEXT DEFAULT 'all' CHECK (access_level IN ('public', 'internal', 'confidential', 'restricted')),
    language TEXT DEFAULT 'en',
    effective_date DATE,
    expiry_date DATE,
    view_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (author_id) REFERENCES users (id),
    FOREIGN KEY (reviewer_id) REFERENCES users (id)
);

-- Create task history table for audit trail
CREATE TABLE task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    field_changed TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by INTEGER,
    change_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks (id),
    FOREIGN KEY (changed_by) REFERENCES users (id)
);

-- Create system logs table
CREATE TABLE system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level TEXT NOT NULL CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    component TEXT, -- 'hrm', 'web_scraper', 'database', etc.
    user_id INTEGER,
    session_id TEXT,
    ip_address TEXT,
    user_agent TEXT,
    execution_time DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Insert sample departments
INSERT INTO departments (name, description, budget) VALUES
('Engineering', 'Software development and technical infrastructure', 2500000.00),
('Marketing', 'Marketing, branding and customer acquisition', 800000.00),
('Sales', 'Sales operations and business development', 1200000.00),
('Human Resources', 'Human resources and talent management', 400000.00),
('Support', 'Customer support and technical assistance', 600000.00),
('Finance', 'Financial planning and accounting', 300000.00),
('Operations', 'Business operations and process management', 500000.00),
('Product', 'Product management and strategy', 900000.00),
('Data Science', 'Data analytics and machine learning', 1500000.00),
('Security', 'Information security and compliance', 700000.00);

-- Insert sample users
INSERT INTO users (employee_id, username, email, first_name, last_name, department_id, position, level, salary, hire_date, phone, skills) VALUES
('EMP001', 'alice.johnson', 'alice.johnson@company.com', 'Alice', 'Johnson', 1, 'Senior Software Engineer', 'Senior', 125000.00, '2022-01-15', '+1-555-0101', '["Python", "JavaScript", "React", "Node.js", "AWS"]'),
('EMP002', 'bob.smith', 'bob.smith@company.com', 'Bob', 'Smith', 2, 'Marketing Manager', 'Manager', 95000.00, '2021-06-20', '+1-555-0102', '["Digital Marketing", "SEO", "Google Analytics", "Content Strategy"]'),
('EMP003', 'carol.davis', 'carol.davis@company.com', 'Carol', 'Davis', 3, 'Sales Representative', 'Mid', 75000.00, '2023-03-10', '+1-555-0103', '["Sales", "CRM", "Negotiation", "Lead Generation"]'),
('EMP004', 'david.wilson', 'david.wilson@company.com', 'David', 'Wilson', 5, 'Senior Support Specialist', 'Senior', 68000.00, '2022-08-05', '+1-555-0104', '["Technical Support", "Troubleshooting", "Customer Service", "SQL"]'),
('EMP005', 'eva.martinez', 'eva.martinez@company.com', 'Eva', 'Martinez', 1, 'DevOps Engineer', 'Senior', 115000.00, '2021-11-12', '+1-555-0105', '["Docker", "Kubernetes", "AWS", "CI/CD", "Infrastructure"]'),
('EMP006', 'frank.brown', 'frank.brown@company.com', 'Frank', 'Brown', 4, 'HR Coordinator', 'Mid', 65000.00, '2020-04-18', '+1-555-0106', '["Recruitment", "Employee Relations", "HR Policies", "Training"]'),
('EMP007', 'grace.lee', 'grace.lee@company.com', 'Grace', 'Lee', 9, 'Data Scientist', 'Senior', 135000.00, '2022-09-01', '+1-555-0107', '["Machine Learning", "Python", "SQL", "Statistics", "Deep Learning"]'),
('EMP008', 'henry.taylor', 'henry.taylor@company.com', 'Henry', 'Taylor', 8, 'Product Manager', 'Lead', 140000.00, '2021-03-22', '+1-555-0108', '["Product Strategy", "Agile", "User Research", "Analytics"]'),
('EMP009', 'iris.chen', 'iris.chen@company.com', 'Iris', 'Chen', 10, 'Security Analyst', 'Mid', 98000.00, '2023-01-16', '+1-555-0109', '["Cybersecurity", "Penetration Testing", "Risk Assessment", "Compliance"]'),
('EMP010', 'jack.robinson', 'jack.robinson@company.com', 'Jack', 'Robinson', 1, 'Frontend Developer', 'Junior', 85000.00, '2023-07-10', '+1-555-0110', '["React", "Vue.js", "CSS", "HTML", "JavaScript"]');

-- Insert sample projects
INSERT INTO projects (project_code, name, description, status, priority, budget, start_date, end_date, manager_id, department_id, client_name, progress_percentage) VALUES
('PRJ-2024-001', 'Customer Portal Redesign', 'Complete redesign of customer-facing portal with modern UI/UX', 'active', 2, 350000.00, '2024-01-01', '2024-06-30', 1, 1, 'Internal', 65),
('PRJ-2024-002', 'Mobile App Development', 'Native iOS and Android app for customer engagement', 'active', 1, 750000.00, '2024-02-01', '2024-12-31', 8, 1, 'External Client A', 35),
('PRJ-2024-003', 'Data Analytics Platform', 'Advanced analytics platform for business intelligence', 'planning', 2, 500000.00, '2024-04-01', '2024-10-31', 7, 9, 'Internal', 10),
('PRJ-2023-004', 'Legacy System Migration', 'Migration from legacy systems to cloud infrastructure', 'completed', 1, 420000.00, '2023-03-01', '2023-12-31', 5, 1, 'Internal', 100),
('PRJ-2024-005', 'Marketing Automation', 'Implementation of marketing automation platform', 'active', 3, 180000.00, '2024-01-15', '2024-05-15', 2, 2, 'Internal', 80),
('PRJ-2024-006', 'Security Compliance Audit', 'Comprehensive security audit and compliance implementation', 'active', 1, 250000.00, '2024-03-01', '2024-08-31', 9, 10, 'External Auditor', 45),
('PRJ-2024-007', 'AI Chatbot Integration', 'Customer service AI chatbot implementation', 'planning', 2, 200000.00, '2024-05-01', '2024-09-30', 7, 9, 'Internal', 5);

-- Insert sample tasks
INSERT INTO tasks (task_code, title, description, status, priority, type, project_id, assigned_to, created_by, estimated_hours, actual_hours, due_date, tags) VALUES
('TSK-001', 'UI/UX Design Review', 'Review and finalize new portal design mockups', 'completed', 2, 'design', 1, 10, 1, 40.0, 38.5, '2024-02-15', '["design", "ui", "review"]'),
('TSK-002', 'Backend API Development', 'Develop REST APIs for customer portal', 'in_progress', 1, 'development', 1, 1, 1, 120.0, 85.0, '2024-03-30', '["api", "backend", "development"]'),
('TSK-003', 'Database Schema Design', 'Design and implement new database schema', 'completed', 2, 'development', 1, 5, 1, 32.0, 29.0, '2024-02-28', '["database", "schema", "design"]'),
('TSK-004', 'Mobile App Wireframes', 'Create wireframes for iOS and Android apps', 'in_progress', 2, 'design', 2, 10, 8, 60.0, 35.0, '2024-03-15', '["mobile", "wireframes", "design"]'),
('TSK-005', 'iOS Development Setup', 'Set up iOS development environment and basic app structure', 'pending', 1, 'development', 2, 1, 8, 80.0, 0.0, '2024-04-01', '["ios", "setup", "development"]'),
('TSK-006', 'Data Pipeline Architecture', 'Design data pipeline architecture for analytics platform', 'pending', 2, 'research', 3, 7, 7, 50.0, 0.0, '2024-04-15', '["data", "pipeline", "architecture"]'),
('TSK-007', 'Marketing Campaign Analysis', 'Analyze Q1 marketing campaign performance', 'completed', 3, 'research', 5, 2, 2, 24.0, 26.0, '2024-04-10', '["marketing", "analysis", "campaign"]'),
('TSK-008', 'Security Vulnerability Assessment', 'Conduct comprehensive security vulnerability assessment', 'in_progress', 1, 'testing', 6, 9, 9, 100.0, 45.0, '2024-05-30', '["security", "vulnerability", "assessment"]'),
('TSK-009', 'Customer Support Training', 'Train support team on new portal features', 'pending', 3, 'training', 1, 4, 1, 16.0, 0.0, '2024-06-15', '["training", "support", "portal"]'),
('TSK-010', 'Performance Optimization', 'Optimize portal performance and loading times', 'pending', 2, 'development', 1, 5, 1, 40.0, 0.0, '2024-05-31', '["performance", "optimization"]');

-- Insert project assignments
INSERT INTO project_assignments (project_id, user_id, role, allocation_percentage, start_date) VALUES
(1, 1, 'lead', 80, '2024-01-01'),
(1, 10, 'member', 100, '2024-01-01'),
(1, 5, 'member', 60, '2024-01-01'),
(2, 8, 'manager', 70, '2024-02-01'),
(2, 1, 'lead', 90, '2024-02-01'),
(2, 10, 'member', 80, '2024-02-01'),
(3, 7, 'manager', 100, '2024-04-01'),
(3, 1, 'member', 40, '2024-04-01'),
(5, 2, 'manager', 90, '2024-01-15'),
(6, 9, 'manager', 100, '2024-03-01'),
(6, 5, 'member', 30, '2024-03-01');

-- Insert knowledge documents
INSERT INTO knowledge_documents (title, content, category, subcategory, version, author_id, tags, access_level, effective_date) VALUES
('Information Security Policy', 
'Our comprehensive information security policy ensures the protection of company and customer data through the following requirements:

1. PASSWORD REQUIREMENTS
- Minimum 12 characters length
- Must include uppercase, lowercase, numbers, and special characters
- Cannot reuse last 12 passwords
- Must be changed every 90 days
- Cannot contain personal information

2. TWO-FACTOR AUTHENTICATION (2FA)
- Required for all business systems and applications
- Use company-approved authenticator apps
- Backup codes must be stored securely
- Report lost devices immediately

3. REMOTE WORK SECURITY
- VPN connection mandatory for all remote access
- Use company-provided devices only
- Secure home Wi-Fi networks (WPA3 encryption)
- Physical workspace security and privacy

4. DATA HANDLING
- Encrypt sensitive data at rest and in transit
- Follow data classification guidelines
- Secure disposal of confidential information
- Regular data backup and recovery testing

5. INCIDENT REPORTING
- Report security incidents within 1 hour
- Contact security team immediately
- Document all incident details
- Follow incident response procedures

6. TRAINING AND AWARENESS
- Annual security training mandatory
- Phishing simulation exercises
- Security updates and communications
- Regular security policy reviews

For questions or clarifications, contact the Information Security team at security@company.com',
'policy', 'security', '3.2', 9, '["security", "policy", "passwords", "2FA", "remote work", "data protection"]', 'internal', '2024-01-01'),

('Remote Work Policy',
'Our remote work policy enables flexible work arrangements while maintaining productivity and security:

1. ELIGIBILITY
- Manager approval required
- Performance standards must be met
- Position suitable for remote work
- Probationary period completion

2. EQUIPMENT AND TECHNOLOGY
- Company-provided laptop and peripherals
- Secure VPN access required
- Business phone line or allowance
- Ergonomic workspace setup

3. WORK ARRANGEMENTS
- Core hours: 9 AM - 3 PM (local time)
- Flexible start/end times within business hours
- Available during scheduled meetings
- Regular communication with team

4. PRODUCTIVITY EXPECTATIONS
- Maintain same performance standards
- Meet all deadlines and deliverables
- Participate actively in team meetings
- Complete time tracking accurately

5. COMMUNICATION REQUIREMENTS
- Daily check-ins with immediate supervisor
- Use company communication tools
- Respond to messages within 4 hours
- Weekly team meetings attendance

6. HOME OFFICE REQUIREMENTS
- Dedicated workspace
- Reliable internet connection (minimum 25 Mbps)
- Quiet environment for calls
- Proper lighting and ventilation

7. SECURITY MEASURES
- Follow all IT security policies
- Secure physical workspace
- Lock devices when not in use
- No family/friends access to work materials

Review and approval process:
- Submit remote work request form
- Manager evaluation and approval
- IT security assessment
- Trial period (30-90 days)
- Ongoing performance monitoring

For remote work requests, contact HR at hr@company.com',
'policy', 'hr', '2.3', 6, '["remote work", "policy", "flexibility", "productivity", "security", "equipment"]', 'internal', '2024-01-01'),

('AI Development Guidelines',
'Guidelines for responsible AI development and deployment within our organization:

1. DATA QUALITY AND ETHICS
- Ensure high-quality, representative training data
- Address bias in datasets and algorithms
- Protect individual privacy and consent
- Regular data quality audits

2. MODEL DEVELOPMENT
- Document model architecture and decisions
- Implement explainable AI principles
- Test for fairness across demographic groups
- Validate model performance rigorously

3. DEPLOYMENT AND MONITORING
- Gradual rollout with monitoring
- Continuous performance evaluation
- Human oversight and intervention capabilities
- Regular model updates and retraining

4. TRANSPARENCY AND ACCOUNTABILITY
- Clear documentation of AI capabilities
- Transparent communication about limitations
- Accountability for AI decisions
- Regular stakeholder reviews

5. PRIVACY AND SECURITY
- Data minimization principles
- Secure model storage and access
- Privacy-preserving techniques
- Compliance with data protection regulations

6. TESTING AND VALIDATION
- Comprehensive testing protocols
- Edge case identification and handling
- Performance metrics and benchmarks
- Independent validation where appropriate

For AI project approval and guidance, contact the AI Ethics Committee at ai-ethics@company.com',
'guideline', 'technical', '2.1', 7, '["ai", "ethics", "development", "machine learning", "guidelines", "responsible ai"]', 'internal', '2024-01-01'),

('Customer Support FAQ',
'Frequently Asked Questions for Customer Support Team:

Q: How do I reset a customer password?
A: Use the admin portal password reset function. Verify customer identity first using our security questions protocol.

Q: What are our standard response times?
A: Critical issues: 1 hour, High priority: 4 hours, Normal: 24 hours, Low priority: 72 hours.

Q: How do I escalate a technical issue?
A: Escalate to Level 2 support through the ticketing system. Include detailed reproduction steps and customer impact assessment.

Q: What information do I need for billing inquiries?
A: Account number, billing period, and detailed description of the inquiry. Route complex billing issues to the Accounts team.

Q: How do I handle feature requests?
A: Log in the feature request system with customer priority and business justification. Forward to Product team for evaluation.

Q: What are our refund policies?
A: Standard refund period is 30 days for software, 14 days for services. Manager approval required for exceptions.

For additional support procedures, see the Customer Support Handbook in the knowledge base.',
'faq', 'support', '1.5', 4, '["support", "faq", "customer service", "procedures", "escalation"]', 'internal', '2024-01-01'),

('Database Backup Procedures',
'Standard procedures for database backup and recovery:

1. BACKUP SCHEDULE
- Full backup: Daily at 2 AM
- Incremental backup: Every 4 hours
- Transaction log backup: Every 15 minutes
- Monthly archive to cold storage

2. BACKUP VERIFICATION
- Daily backup integrity checks
- Weekly restore testing
- Monthly full recovery drill
- Document all test results

3. RECOVERY PROCEDURES
- Assess data loss scope
- Identify recovery point objective
- Execute appropriate recovery strategy
- Validate data integrity post-recovery

4. MONITORING AND ALERTS
- Automated backup monitoring
- Immediate alert on backup failures
- Disk space monitoring
- Performance impact assessment

Contact the Database Team for backup-related issues: db-admin@company.com',
'procedure', 'technical', '1.8', 5, '["database", "backup", "recovery", "procedures", "monitoring"]', 'confidential', '2024-01-01');

-- Create indexes for better performance
CREATE INDEX idx_users_department ON users(department_id);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_tasks_project ON tasks(project_id);
CREATE INDEX idx_tasks_assigned ON tasks(assigned_to);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_due_date ON tasks(due_date);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_manager ON projects(manager_id);
CREATE INDEX idx_knowledge_category ON knowledge_documents(category);
CREATE INDEX idx_knowledge_status ON knowledge_documents(status);
CREATE INDEX idx_system_logs_level ON system_logs(log_level);
CREATE INDEX idx_system_logs_component ON system_logs(component);
CREATE INDEX idx_system_logs_created ON system_logs(created_at);

-- Create triggers for updated_at timestamps
CREATE TRIGGER update_users_timestamp 
    AFTER UPDATE ON users
    BEGIN
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER update_projects_timestamp 
    AFTER UPDATE ON projects
    BEGIN
        UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER update_tasks_timestamp 
    AFTER UPDATE ON tasks
    BEGIN
        UPDATE tasks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

CREATE TRIGGER update_knowledge_timestamp 
    AFTER UPDATE ON knowledge_documents
    BEGIN
        UPDATE knowledge_documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;

-- Create views for common queries
CREATE VIEW active_projects AS
SELECT 
    p.*,
    u.full_name as manager_name,
    d.name as department_name
FROM projects p
LEFT JOIN users u ON p.manager_id = u.id
LEFT JOIN departments d ON p.department_id = d.id
WHERE p.status IN ('planning', 'active');

CREATE VIEW task_summary AS
SELECT 
    t.*,
    assigned_user.full_name as assigned_to_name,
    creator.full_name as created_by_name,
    p.name as project_name
FROM tasks t
LEFT JOIN users assigned_user ON t.assigned_to = assigned_user.id
LEFT JOIN users creator ON t.created_by = creator.id
LEFT JOIN projects p ON t.project_id = p.id;

CREATE VIEW user_workload AS
SELECT 
    u.id,
    u.full_name,
    u.department_id,
    d.name as department_name,
    COUNT(t.id) as total_tasks,
    COUNT(CASE WHEN t.status = 'in_progress' THEN 1 END) as active_tasks,
    COUNT(CASE WHEN t.status = 'pending' THEN 1 END) as pending_tasks,
    SUM(CASE WHEN t.status != 'completed' THEN t.estimated_hours ELSE 0 END) as pending_hours
FROM users u
LEFT JOIN departments d ON u.department_id = d.id
LEFT JOIN tasks t ON u.id = t.assigned_to
WHERE u.status = 'active'
GROUP BY u.id, u.full_name, u.department_id, d.name;

-- Insert initial system log
INSERT INTO system_logs (log_level, message, component) VALUES
('INFO', 'Database schema created and initial data loaded successfully', 'database');

-- Display summary
SELECT 'Database Setup Complete' as status;
SELECT 'Departments: ' || COUNT(*) as summary FROM departments
UNION ALL
SELECT 'Users: ' || COUNT(*) FROM users
UNION ALL  
SELECT 'Projects: ' || COUNT(*) FROM projects
UNION ALL
SELECT 'Tasks: ' || COUNT(*) FROM tasks
UNION ALL
SELECT 'Knowledge Documents: ' || COUNT(*) FROM knowledge_documents;