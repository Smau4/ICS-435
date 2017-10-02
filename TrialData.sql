CREATE TABLE IF NOT EXISTS TrialData (
                                        point_id integer PRIMARY KEY,
                                        trial_id integer NOT NULL,
                                        x_point real NOT NULL,
                                        y_point real NOT NULL,
                                        label integer NOT NULL,
                                        FOREIGN KEY (trial_id) REFERENCES Trials (trial_id)
                                    );