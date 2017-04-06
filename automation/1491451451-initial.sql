CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE photos(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    parent_url TEXT,
    sha256 TEXT NOT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT current_timestamp,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT current_timestamp,
    UNIQUE(sha256, url)
);

CREATE TABLE faces(
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photo_id UUID NOT NULL REFERENCES photos(id),
    feature_vector NUMERIC[4] NOT NULL,

    top_left_x INT NOT NULL,
    top_left_y INT NOT NULL,
    bottom_right_x INT NOT NULL,
    bottom_right_y INT NOT NULL,

    UNIQUE(photo_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
);
